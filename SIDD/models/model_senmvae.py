import os
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel  # , DistributedDataParallel
from collections import OrderedDict
from torch.optim import Adam
from torch.optim import lr_scheduler
import lpips
from .discriminator import GANLoss


def nl_loss_label_func(label):
    return (label == -1).float().squeeze().bool()

def rec_H_loss_label_func(label):
    return ((label == -1).float() + (label == 0).float()).squeeze().bool()

def rec_L_loss_label_func(label):
    return ((label == -1).float() + (label == 1).float()).squeeze().bool()

def degrade_level_process(sigma, sigma_pred, label):
    return sigma * (label == -1).float() + sigma_pred * (label == 1).float()


class SeNMVAEIR():
    def __init__(self, opt):
        self.opt = opt                         # opt
        self.save_dir = opt['path']['models']  # save models
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']        # training or not
        self.schedulers = []                   # schedulers
        
        self.model = self.define_net().to(self.device)
        self.model = DataParallel(self.model)
        
        self.noise_level_net = self.define_nlnet().to(self.device)
        self.noise_level_net = DataParallel(self.noise_level_net)

        self.netD = self.define_D().to(self.device)
        self.netD = DataParallel(self.netD)

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    def init_train(self):
        self.opt_train = self.opt['train']    # training option
        self.load()                           # load model
        self.model.train()
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log 


    def load(self):
        load_path = self.opt['path']['pretrained_net']
        if load_path is not None:
            print('Loading model [{:s}] ...'.format(load_path))
            self.load_network(load_path, self.model)


    def save(self, iter_label):
        self.save_network(self.save_dir, self.model, iter_label)

    def define_loss(self):
        self.lossfn = nn.MSELoss(reduction='mean').to(self.device)
        self.losslpips = lpips.LPIPS(net='alex').to(self.device)
        
        self.D_lossfn = GANLoss('ragan', 1.0, 0.0).to(self.device)
        self.D_lossfn_weight = self.opt_train['D_lossfn_weight']

        self.D_update_ratio = self.opt_train['D_update_ratio'] if self.opt_train['D_update_ratio'] else 1
        self.D_init_iters = self.opt_train['D_init_iters'] if self.opt_train['D_init_iters'] else 0

    def define_optimizer(self):
        optim_params = [p for p in self.model.parameters()] + [p for p in self.noise_level_net.parameters()]
        self.optimizer = Adam(optim_params, lr=self.opt_train['optimizer_lr'], weight_decay=0)
        self.D_optimizer = Adam(self.netD.parameters(), lr=self.opt_train['D_optimizer_lr'], weight_decay=0)


    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.optimizer,
                                                        self.opt_train['scheduler_milestones'],
                                                        self.opt_train['scheduler_gamma']
                                                        ))
        self.schedulers.append(lr_scheduler.MultiStepLR(self.D_optimizer,
                                                        self.opt_train['D_scheduler_milestones'],
                                                        self.opt_train['D_scheduler_gamma']
                                                        ))


    def feed_data(self, data):
        self.img_H = data['img_H'].to(self.device)
        self.img_L = data['img_L'].to(self.device)
        self.label = data['label'].to(self.device)
        

    def optimize_parameters(self, current_step):
        # ------------------------------------
        # optimize G
        # ------------------------------------
        for p in self.netD.parameters():
            p.requires_grad = False

        self.optimizer.zero_grad()

        if self.opt_train['KL_anneal'] == 'linear':
            coeff = min((current_step / self.opt_train['KL_anneal_maxiter']), 1)
            kln_weight = coeff * self.opt_train['KL_weight']
        else:
            kln_weight = self.opt_train['KL_weight']

        noise = self.img_L - self.img_H
        sigma = torch.std(noise, dim=(1, 2, 3), keepdim=True)
        sigma_pred = self.noise_level_net(self.img_L)
        nl_loss = torch.abs(sigma - sigma_pred).mean(dim=(1, 2, 3)) 
        nl_mask = nl_loss_label_func(self.label)
        nl_loss = nl_loss[nl_mask].mean()
        
        degrade_level = degrade_level_process(sigma, sigma_pred, self.label)

        rec_H, rec_L, klz_loss, kln_loss = self.model(self.img_H, self.img_L, degrade_level, self.label)

        rec_L_mask = rec_L_loss_label_func(self.label).bool()
        rec_H_mask = rec_H_loss_label_func(self.label).bool()

        if current_step % self.D_update_ratio == 0 and current_step > self.D_init_iters:  # updata D first

            rec_H_loss = self.lossfn(rec_H[rec_H_mask], self.img_H[rec_H_mask])
            rec_L_loss = self.lossfn(rec_L[rec_L_mask], self.img_L[rec_L_mask])

            rec_loss = rec_H_loss + rec_L_loss

            per_loss = self.losslpips(rec_L[rec_L_mask], self.img_L[rec_L_mask]).mean()

            pred_g_fake = self.netD(rec_L[rec_L_mask])

            pred_d_real = self.netD(self.img_L[rec_L_mask]).detach()
            D_loss = self.D_lossfn_weight * (
                self.D_lossfn(pred_d_real - torch.mean(pred_g_fake), False) +
                self.D_lossfn(pred_g_fake - torch.mean(pred_d_real), True)) / 2
            
            gan_loss = D_loss  # 3) GAN loss
            
            loss1 = rec_loss.mean().unsqueeze(0)
            loss2 = kln_weight * kln_loss.mean().unsqueeze(0)
            loss3 = kln_weight * self.opt_train['KLZ_weight'] * klz_loss.mean().unsqueeze(0)
            loss4 = nl_loss.mean().unsqueeze(0)
            loss5 = self.opt_train['Per_weight'] * per_loss.mean().unsqueeze(0)
            loss6 = gan_loss.mean().unsqueeze(0)
            
            losses = torch.cat([loss1, loss2, loss3, loss4, loss5, loss6], dim=0)
            loss_nan_mask = torch.isnan(losses)
            loss = losses[~loss_nan_mask].sum()
            
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 200).item()
            
            if grad_norm < 150:
                self.optimizer.step()

        # ------------------------------------
        # optimize D
        # ------------------------------------
        for p in self.netD.parameters():
            p.requires_grad = True

        self.D_optimizer.zero_grad()
        loss_D_total = 0

        pred_d_real = self.netD(self.img_L[rec_L_mask])           # 1) real data
        pred_d_fake = self.netD(rec_L[rec_L_mask].detach())       # 2) fake data, detach to avoid BP to G

        l_d_real = self.D_lossfn(pred_d_real - torch.mean(pred_d_fake), True)
        l_d_fake = self.D_lossfn(pred_d_fake - torch.mean(pred_d_real), False)
        loss_D_total = (l_d_real + l_d_fake) / 2
        
        loss_D_total.backward()
        
        loss_nan = torch.any(torch.isnan(loss_D_total))
        
        if loss_nan == 0:
            self.D_optimizer.step()

        if current_step % self.D_update_ratio == 0 and current_step > self.D_init_iters:
            self.log_dict['loss'] = loss.item()
            self.log_dict['reconstruction_loss'] = loss1.item()
            self.log_dict['KL_loss'] = loss2.item()
            self.log_dict['KLZ_loss'] = loss3.item()
            self.log_dict['NL_loss'] = loss4.item()
            self.log_dict['Per_loss'] = loss5.item()
            self.log_dict['D_loss'] = loss6.item()

        self.log_dict['l_d_real'] = l_d_real.item()  
        self.log_dict['l_d_fake'] = l_d_fake.item()  
        self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
        self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())

    def test(self):
        self.model.eval()
        
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model
        
        with torch.no_grad():
            self.img_G = model.translate(self.img_H)
            self.img_E = model.denoise(self.img_L)

        self.model.train()


    def current_log(self):
        return self.log_dict


    def current_visuals(self):
        out_dict = OrderedDict()
        out_dict['img_H'] = self.img_H.detach()[0].float().cpu()
        out_dict['img_L'] = self.img_L.detach()[0].float().cpu()
        out_dict['img_G'] = self.img_G.detach()[0].float().cpu()
        out_dict['img_E'] = self.img_E.detach()[0].float().cpu()
        return out_dict


    def current_results(self):
        out_dict = OrderedDict()
        out_dict['img_H'] = self.img_H.detach()[0].float().cpu()
        out_dict['img_L'] = self.img_L.detach()[0].float().cpu()
        out_dict['img_G'] = self.img_G.detach()[0].float().cpu()
        out_dict['img_E'] = self.img_E.detach()[0].float().cpu()
        return out_dict


    def update_learning_rate(self, n):
        for scheduler in self.schedulers:
            scheduler.step()


    def current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]


    """
    # ----------------------------------------
    # Information of net
    # ----------------------------------------
    """

    def print_network(self):
        msg = self.describe_network(self.model)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.model)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.model)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.model)
        return msg

    # ----------------------------------------
    # network name and number of parameters
    # ----------------------------------------
    
    def describe_network(self, model):
        if isinstance(model, nn.DataParallel):
            model = model.module
            
        msg = '\n'
        msg += 'Networks name: {}'.format(model.__class__.__name__) + '\n'
        msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), model.parameters()))) + '\n'
        msg += 'Net structure:\n{}'.format(str(model)) + '\n'
        
        return msg

    # ----------------------------------------
    # parameters description
    # ----------------------------------------
    def describe_params(self, model):
        if isinstance(model, nn.DataParallel):
            model = model.module
            
        msg = '\n'
        msg += ' | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}'.format('mean', 'min', 'max', 'std', 'shape', 'param_name') + '\n'
        for name, param in model.state_dict().items():
            if not 'num_batches_tracked' in name:
                v = param.data.clone().float()
                msg += ' | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} | {} || {:s}'.format(v.mean(), v.min(), v.max(), v.std(), v.shape, name) + '\n'
        
        return msg

    """
    # ----------------------------------------
    # Save prameters
    # Load prameters
    # ----------------------------------------
    """
    # ----------------------------------------
    # save the state_dict of the network
    # ----------------------------------------
    def save_network(self, save_dir, model, iter_label):
        save_filename = '{}.pth'.format(iter_label)
        save_path = os.path.join(save_dir, save_filename)
        if isinstance(model, nn.DataParallel):
            model = model.module
            
        model_state_dict = model.state_dict()
        
        for key, param in model_state_dict.items():
            model_state_dict[key] = param.cpu()
        
        states = model_state_dict
        torch.save(states, save_path)


    # ----------------------------------------
    # load the state_dict of the network
    # ----------------------------------------
    def load_network(self, load_path, model, strict=True):
        if isinstance(model, nn.DataParallel):
            model = model.module
            
        states = torch.load(load_path)
        model.load_state_dict(states, strict=strict)

    def define_net(self):
        from SIDD.models.network_senmvae import SeNMVAE
        net = SeNMVAE(in_ch = self.opt['n_channels'], 
                      nls = self.opt['model']['nls'], 
                      zn_dim = self.opt['model']['zn_dim'], 
                      num_down = self.opt['model']['num_down'])
        return net

    def define_nlnet(self):
        from models.network_nlnet import NoiseLevelNet
        net = NoiseLevelNet(self.opt['n_channels'])
        return net

    def define_D(self):
        from models.discriminator import NLayerDiscriminator
        net = NLayerDiscriminator(input_nc=self.opt['n_channels'])
        return net


















