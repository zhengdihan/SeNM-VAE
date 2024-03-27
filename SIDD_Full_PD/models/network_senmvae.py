import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
    return -0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)

def draw_gaussian_diag_samples(mu, logsigma):
    eps = torch.empty_like(mu).normal_(0., 1.)
    return torch.exp(logsigma) * eps + mu

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

class Resize(nn.Module):
    def __init__(self, in_channels, out_channels, sf):
        super().__init__()
        
        self.sf = sf
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
                
    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=self.sf, mode='bilinear')
        
        return x

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
    def __init__(self, in_channels, out_channels, last_scale=0.1):
        super().__init__()
        
        self.m = nn.Sequential(
                 BasicBlock(in_channels),
                 nn.Conv2d(in_channels, out_channels, 3, 1, 1)
                 )
        
        self.v = nn.Sequential(
                 BasicBlock(in_channels),
                 nn.Conv2d(in_channels, out_channels, 3, 1, 1)
                 )
        
        self.m[1].weight.data *= last_scale
        self.v[1].weight.data *= last_scale
        
    def forward(self, x):
        return self.m(x), self.v(x)

class SeNMVAE(nn.Module):
    def __init__(self, in_ch=3, nf=64, nls=[16, 16]):
        super(SeNMVAE, self).__init__()
        
        zh_dim = int(nf * 0.25)
        
        self.inconv_H = nn.Conv2d(in_ch, nf, 3, 1, 1)
        self.inconv_L = nn.Conv2d(in_ch + 1, nf, 3, 1, 1)
        
        encs_H = []
        encs_L = []
        Gauconv_q = []
        Gauconv_p = []
        dec_z = []
        dec_zh = []
        proj_zh = []
        proj_z = []
        
        for i, nl in enumerate(nls):

            for _ in range(nl):
                encs_H.append(BasicBlock(nf))
                encs_L.append(BasicBlock(nf))
                Gauconv_q.append(GauBlock(nf, zh_dim))
                Gauconv_p.append(GauBlock(nf, zh_dim))
                dec_z.append(BasicBlock(nf))
                dec_zh.append(BasicBlock(nf))
                proj_zh.append(nn.Conv2d(zh_dim, nf, 1))
                proj_z.append(nn.Conv2d(nf, nf, 1))
                
            if i != len(nls) - 1:
                Gauconv_q.append(GauBlock(nf, zh_dim))
                Gauconv_p.append(GauBlock(nf, zh_dim))
                encs_H.append(Resize(nf, nf, 1/2))
                encs_L.append(Resize(nf, nf, 1/2))
                dec_z.append(BasicBlock(nf))
                dec_zh.append(Resize(nf, nf, sf=2))
                proj_zh.append(nn.Conv2d(zh_dim, nf, 1))
                proj_z.append(nn.Conv2d(nf, nf, 1))
                
        self.encs_H = nn.ModuleList(encs_H)
        self.encs_L = nn.ModuleList(encs_L)
        self.Gauconv_q = nn.ModuleList(Gauconv_q)
        self.Gauconv_p = nn.ModuleList(Gauconv_p)
        self.dec_z = nn.ModuleList(dec_z)
        self.dec_zh = nn.ModuleList(dec_zh)
        
        initialize_weights(proj_zh, 0.1)
        self.proj_zh = nn.ModuleList(proj_zh)
        initialize_weights(proj_z, 0.1)
        self.proj_z = nn.ModuleList(proj_z)
                
        self.outconv = nn.Conv2d(nf, in_ch, 3, 1, 1)        
        self.nls = nls


    def forward(self, img_H, img_L):
        b, c, h, w = img_H.shape

        noise_level = torch.std(img_H - img_L, dim=(1, 2, 3), keepdim=True) * torch.ones(b, 1, h, w).to(img_L.device)
        img_L_with_nl = torch.cat((img_L, noise_level), dim=1)

        acts_H, acts_L = self.encode(img_H, img_L_with_nl)
        dec_H, kl_loss = self.decode(acts_H, acts_L)
        
        kl_loss = kl_loss / (b*c*h*w)
        kl_loss = kl_loss.unsqueeze(0)
        
        return dec_H, kl_loss


    def encode(self, img_H, img_L):            
        conv_H = self.inconv_H(img_H)
        conv_L = self.inconv_L(img_L)
        
        acts_H = {}
        acts_L = {}
        
        act_H = conv_H
        act_L = conv_L
        
        for i in range(len(self.encs_H)):
            enc_H = self.encs_H[i]
            enc_L = self.encs_L[i]
            
            act_H = enc_H(act_H)
            act_L = enc_L(act_L)
            
            acts_H[i] = act_H
            acts_L[i] = act_L
        
        return acts_H, acts_L
    
    
    def decode(self, acts_H, acts_L):
        
        kls = 0
        dec = 0
        
        for i in range(len(acts_H))[::-1]:
            act_H = acts_H[i]
            act_L = acts_L[i]

            enc_z = act_L + dec
            dec = dec + self.proj_z[i](enc_z)

            dec = self.dec_z[i](dec)

            qm, qv = self.Gauconv_q[i](dec + act_H)
            pm, pv = self.Gauconv_p[i](dec)
            
            enc_h = draw_gaussian_diag_samples(qm, qv)
            kl = gaussian_analytical_kl(qm, pm, qv, pv)
            
            dec = dec + self.proj_zh[i](enc_h)
            dec = self.dec_zh[i](dec)
                
            kls = kls + kl.sum()
        
        dec = self.outconv(dec)
        
        return dec, kls

    def decode_uncond(self, acts_L, temperature=1.0):
        
        dec = 0
        
        for i in range(len(acts_L))[::-1]:

            act_L = acts_L[i]
            
            enc_z = act_L + dec
            dec = dec + self.proj_z[i](enc_z)

            dec = self.dec_z[i](dec)            
            
            pm, pv = self.Gauconv_p[i](dec)
            enc_h = draw_gaussian_diag_samples(pm, pv) * temperature

            dec = dec + self.proj_zh[i](enc_h)
            dec = self.dec_zh[i](dec)
        
        dec = self.outconv(dec)
        
        return dec

    def encode_uncond(self, img_L):
        conv_L = self.inconv_L(img_L)
        
        acts_L = {}
        act_L = conv_L
        
        for i in range(len(self.encs_L)):
            enc_L = self.encs_L[i]
            act_L = enc_L(act_L)
            acts_L[i] = act_L
        
        return acts_L

    def generate(self, img_L, noise_level=25/255, temperature=1.0):
        b, c, h, w= img_L.shape

        noise_level = noise_level * torch.ones(b, 1, h, w).to(img_L.device)
        img_L_with_nl = torch.cat((img_L, noise_level), dim=1)

        acts_L = self.encode_uncond(img_L_with_nl)
        dec = self.decode_uncond(acts_L, temperature)
        
        return dec

