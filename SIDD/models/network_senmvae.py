import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init


def klz_loss_label_func(label):
    return (label == -1).float().squeeze().bool()

def kln_loss_label_func(label):
    return ((label == -1).float() + (label == 1).float()).squeeze().bool()

def rec_H_loss_label_func(label):
    return ((label == -1).float() + (label == 0).float()).squeeze().bool()

def rec_L_loss_label_func(label):
    return ((label == -1).float() + (label == 1).float()).squeeze().bool()

def nl_loss_label_func(label):
    return (label == -1).float().squeeze().bool()

def cat_degrade_level(x, degrade_level):
    degrade = torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]))
    degrade = degrade.to(x.device) * degrade_level
    
    return torch.cat([x, degrade], dim=1)

def draw_samples(hm, hv, lm, lv, mode, label):
    enc_z_pd = draw_mix_gaussian_samples(hm, hv, lm, lv, mode)
    enc_z_hd = draw_gaussian_diag_samples(hm, hv)
    enc_z_ld = draw_gaussian_diag_samples(lm, lv)
    
    enc_z = enc_z_pd * (label == -1).float() + enc_z_hd * (label == 0).float() + enc_z_ld * (label == 1).float()
    
    return enc_z

def degrade_level_process(sigma, sigma_pred, label):
    return sigma * (label == -1).float() + sigma_pred * (label == 1).float()

def zero_weight_nets(nets):
    for net in nets:
        net.weight.data *= 0
        
    return nets

def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
    return -0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)

def draw_gaussian_diag_samples(mu, logsigma):
    eps = torch.empty_like(mu).normal_(0., 1.)
    return torch.exp(logsigma) * eps + mu

def draw_mix_gaussian_samples(mu1, logsigma1, mu2, logsigma2, mode):
    z1 = draw_gaussian_diag_samples(mu1, logsigma1)
    z2 = draw_gaussian_diag_samples(mu2, logsigma2)
    z = z1 * mode + z2 * (1 - mode)
    
    return z

def gaussian_analytical_w2(mu1, mu2, logsigma1, logsigma2):
    p1 = torch.pow((mu1 - mu2), 2)
    p2 = torch.pow(logsigma1.exp() - logsigma2.exp(), 2)
    
    return p1 + p2

def gaussian_analytical_js(mu1, mu2, logsigma1, logsigma2):
    return 0.5 * gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2) + 0.5 * gaussian_analytical_kl(mu2, mu1, logsigma2, logsigma1)

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class BasicBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.net = ResidualDenseBlock_5C(in_channels)
        
    def forward(self, x):
        return self.net(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.down = nn.Sequential(
                    nn.PixelUnshuffle(2), 
                    nn.Conv2d(in_channels * 4, out_channels, 3, 1, 1), 
                    )
        
    def forward(self, x):
        return self.down(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.up = nn.Sequential(
                    nn.PixelShuffle(2), 
                    nn.Conv2d(in_channels // 4, out_channels, 3, 1, 1)
                    )
        
    def forward(self, x):
        return self.up(x)

class GauBlock(nn.Module):
    def __init__(self, in_channels, out_channels, zero_last=True):
        super().__init__()
        
        self.m = nn.Sequential(
                 BasicBlock(in_channels),
                 nn.Conv2d(in_channels, out_channels, 1)
                 )
        
        self.v = nn.Sequential(
                 BasicBlock(in_channels),
                 nn.Conv2d(in_channels, out_channels, 1)
                 )
        
        if zero_last:
            self.m[1].weight.data *= 0
            self.v[1].weight.data *= 0
        
    def forward(self, x):
        return self.m(x), self.v(x)


class SeNMVAE(nn.Module):
    def __init__(self, in_ch=3, nf=64, nls=[2, 2], zn_dim=16, num_down=0):
        super(SeNMVAE, self).__init__()
        
        self.inconv_H = nn.Sequential(nn.Conv2d(in_ch, nf, 1), 
                                      *[DownSample(nf, nf) for _ in range(num_down)])
        self.inconv_L = nn.Sequential(nn.Conv2d(in_ch, nf, 1), 
                                      *[DownSample(nf, nf) for _ in range(num_down)])
        self.inconv_N = nn.Sequential(nn.Conv2d(in_ch, nf, 1), 
                                      *[DownSample(nf, nf) for _ in range(num_down)])
        
        encs_H = []
        encs_L = []
        encs_N = []
        Gauconv_zn_q = []
        Gauconv_zn_p = []
        Gauconv_z_H = []
        Gauconv_z_L = []
        dec = []
        proj_zn = []
        proj_z = []
        
        for i, nl in enumerate(nls):

            for _ in range(nl):
                encs_H.append(BasicBlock(nf))
                encs_L.append(BasicBlock(nf))
                encs_N.append(BasicBlock(nf))
                Gauconv_zn_q.append(GauBlock(nf+1, zn_dim))
                Gauconv_zn_p.append(GauBlock(nf+1, zn_dim))
                Gauconv_z_H.append(GauBlock(nf, nf))
                Gauconv_z_L.append(GauBlock(nf, nf))
                dec.append(BasicBlock(nf))
                proj_zn.append(nn.Conv2d(zn_dim, nf, 1))
                proj_z.append(nn.Conv2d(nf, nf, 1))
                
            if i != len(nls) - 1:
                Gauconv_zn_q.append(GauBlock(nf+1, zn_dim))
                Gauconv_zn_p.append(GauBlock(nf+1, zn_dim))
                Gauconv_z_L.append(GauBlock(nf, nf))
                Gauconv_z_H.append(GauBlock(nf, nf))
                encs_H.append(DownSample(nf, nf))
                encs_L.append(DownSample(nf, nf))
                encs_N.append(DownSample(nf, nf))
                dec.append(UpSample(nf, nf))
                proj_zn.append(nn.Conv2d(zn_dim, nf, 1))
                proj_z.append(nn.Conv2d(nf, nf, 1))
        
        self.encs_H = nn.ModuleList(encs_H)
        self.encs_L = nn.ModuleList(encs_L)
        self.encs_N = nn.ModuleList(encs_N)
        self.Gauconv_zn_q = nn.ModuleList(Gauconv_zn_q)
        self.Gauconv_zn_p = nn.ModuleList(Gauconv_zn_p)
        self.Gauconv_z_L = nn.ModuleList(Gauconv_z_L)
        self.Gauconv_z_H = nn.ModuleList(Gauconv_z_H)
        self.dec = nn.ModuleList(dec)
        
        self.proj_zn = nn.ModuleList(proj_zn)
        self.proj_zn = zero_weight_nets(self.proj_zn)
        self.proj_z = nn.ModuleList(proj_z)
        self.proj_z = zero_weight_nets(self.proj_z)
        
        self.outconv = nn.Sequential(*[UpSample(nf, nf) for _ in range(num_down)], 
                                     nn.Conv2d(nf, nf, 3, 1, 1), 
                                     nn.ReLU(True), 
                                     nn.Conv2d(nf, in_ch, 3, 1, 1))
        
        self.nls = nls

    def forward(self, img_H, img_L, degrade_level, label):
                
        acts_H, acts_L, acts_N = self.encode(img_H, img_L)
        
        dec_H, klz_H_loss = self.decode_H(acts_H, acts_L, label)
        dec_L, klz_L_loss, kln_loss = self.decode_L(acts_H, acts_L, acts_N, label, degrade_level)
        
        klz_loss = klz_H_loss + klz_L_loss

        klz_mask = klz_loss_label_func(label)
        klz_loss = klz_loss[klz_mask].mean()

        kln_mask = kln_loss_label_func(label)
        kln_loss = kln_loss[kln_mask].mean()
                        
        klz_loss = klz_loss.unsqueeze(0)
        kln_loss = kln_loss.unsqueeze(0)
        
        return dec_H, dec_L, klz_loss, kln_loss

    def encode(self, img_H, img_L):
        
        conv_H = self.inconv_H(img_H)
        conv_L = self.inconv_L(img_L)
        conv_N = self.inconv_N(img_L)
        
        acts_H = []
        act_H = conv_H
        
        for enc_H in self.encs_H:
            act_H = enc_H(act_H)
            acts_H.append(act_H)
            
        acts_L = []
        act_L = conv_L
        
        for enc_L in self.encs_L:
            act_L = enc_L(act_L)
            acts_L.append(act_L)
            
        acts_N = []
        act_N = conv_N
        
        for enc_N in self.encs_N:
            act_N = enc_N(act_N)
            acts_N.append(act_N)        
        
        return acts_H, acts_L, acts_N
    
    def decode_H(self, acts_H, acts_L, label):
        
        b = label.shape[0]
        
        dec = 0
        klzs = 0
        
        mode = (torch.rand(b, 1, 1, 1) > 0.5).float().to(label.device)
        
        for i in range(len(acts_H))[::-1]:
            act_H, act_L = acts_H[i], acts_L[i]
            
            hm, hv = self.Gauconv_z_H[i](dec + act_H)
            lm, lv = self.Gauconv_z_L[i](dec + act_L)
            
            enc_z = draw_samples(hm, hv, lm, lv, mode, label)
            klz = gaussian_analytical_kl(lm, hm, lv, hv)
            
            dec = dec + self.proj_z[i](enc_z)
            dec = self.dec[i](dec)
            
            klzs = klzs + klz.mean(dim=(1, 2, 3))
            
        dec = self.outconv(dec)
                
        return dec, klzs

    def decode_L(self, acts_H, acts_L, acts_N, label, degrade_level):
        
        b = label.shape[0]
        
        dec = 0
        klns = 0
        klzs = 0
        
        mode = (torch.rand(b, 1, 1, 1) > 0.5).float().to(label.device)
        
        for i in range(len(acts_H))[::-1]:
            
            act_H, act_L, act_N = acts_H[i], acts_L[i], acts_N[i]
            
            hm, hv = self.Gauconv_z_H[i](dec + act_H)
            lm, lv = self.Gauconv_z_L[i](dec + act_L)
            
            enc_z = draw_samples(hm, hv, lm, lv, mode, label)
            klz = gaussian_analytical_kl(lm, hm, lv, hv)
            
            dec = dec + self.proj_z[i](enc_z)
            
            qm, qv = self.Gauconv_zn_q[i](cat_degrade_level(dec + act_N, degrade_level))
            pm, pv = self.Gauconv_zn_p[i](cat_degrade_level(dec, degrade_level))
            
            enc_n = draw_gaussian_diag_samples(qm, qv)
            kln = gaussian_analytical_kl(qm, pm, qv, pv)
            
            dec = dec + self.proj_zn[i](enc_n)
            
            dec = self.dec[i](dec)

            klzs = klzs + klz.mean(dim=(1, 2, 3))
            klns = klns + kln.mean(dim=(1, 2, 3)) 
            
        dec = self.outconv(dec)
        
        return dec, klzs, klns

    def decode_uncond_H(self, acts_H, degrade_level, temperature=1.0):
        
        dec = 0
        
        for i in range(len(acts_H))[::-1]:
            act_H = acts_H[i]
            
            hm, hv = self.Gauconv_z_H[i](dec + act_H)
            enc_z = draw_gaussian_diag_samples(hm, hv)
            
            dec = dec + self.proj_z[i](enc_z)
            
            pm, pv = self.Gauconv_zn_p[i](cat_degrade_level(dec, degrade_level))
            enc_n = draw_gaussian_diag_samples(pm, pv) * temperature

            dec = dec + self.proj_zn[i](enc_n)
            dec = self.dec[i](dec)
        
        dec = self.outconv(dec)
        
        return dec

    def encode_uncond_H(self, img_H):
        conv_H = self.inconv_H(img_H)
        
        acts_H = []
        act_H = conv_H
        
        for enc_H in self.encs_H:
            act_H = enc_H(act_H)
            acts_H.append(act_H)
        
        return acts_H
    
    def encode_uncond_L(self, img_L):    
        conv_L = self.inconv_L(img_L)
        
        acts_L = []
        act_L = conv_L
        
        for enc_L in self.encs_L:
            act_L = enc_L(act_L)
            acts_L.append(act_L)
        
        return acts_L

    def decode_uncond_L(self, acts_L):
        
        dec = 0
        
        for i in range(len(acts_L))[::-1]:
            act_L = acts_L[i]

            lm, lv = self.Gauconv_z_L[i](dec + act_L)            
            enc_z = draw_gaussian_diag_samples(lm, lv)
            dec = dec + self.proj_z[i](enc_z)
            dec = self.dec[i](dec)
                        
        dec = self.outconv(dec)

        return dec

    def translate(self, img_H, degrade_level=25./255, temperature=1.0):
        acts_H = self.encode_uncond_H(img_H)
        dec = self.decode_uncond_H(acts_H, degrade_level, temperature)
        
        return dec

    def denoise(self, img_L):
        acts_L = self.encode_uncond_L(img_L)
        dec = self.decode_uncond_L(acts_L)
        
        return dec

    def reconst_H(self, img_H, img_L):
        label = torch.ones(1, 1, 1, 1).to(img_H.device)
        acts_H, acts_L, acts_N = self.encode(img_H, img_L)
        dec_H, _ = self.decode_H(acts_H, acts_L, label)
        
        return dec_H


# net = PVAE()
# x = torch.rand(2, 3, 32, 32)
# y = torch.rand(2, 3, 32, 32)
# label = torch.ones(2, 1, 1, 1) * -1
# a, b, c, d = net(x, y, label)
# net.translate(x)

# print(a)
# print(b)
# print(c)
# print(d)



