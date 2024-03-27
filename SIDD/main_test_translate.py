import os.path
import logging

import numpy as np
from collections import OrderedDict

import torch
import torch.distributions as dist

from utils import utils_logger
from utils import utils_image as util
from utils.utils_eval import eval_split

from tqdm import tqdm
import random

def random_noise_levels_sidd(bs):
    """ Where read_noise in SIDD is not 0 """
    log_min_shot_noise = torch.log10(torch.Tensor([0.00068674]))
    log_max_shot_noise = torch.log10(torch.Tensor([0.02194856]))
    distribution = dist.uniform.Uniform(log_min_shot_noise, log_max_shot_noise)
      
    log_shot_noise = distribution.sample(torch.Size([bs]))
    shot_noise = torch.pow(10,log_shot_noise)
      
    distribution = dist.normal.Normal(torch.Tensor([0.0]), torch.Tensor([0.20]))
    read_noise = distribution.sample(torch.Size([bs]))
    line = lambda x: 1.85 * x + 0.30  ### Line SIDD test set
    log_read_noise = line(log_shot_noise) + read_noise
    read_noise = torch.pow(10,log_read_noise)
    
    variance = shot_noise + read_noise
    std = torch.sqrt(variance) * 1
    return std.unsqueeze(2).unsqueeze(3)

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    model_name = 'SeNM-VAE_10_paired' # SeNM-VAE_10_paired, SeNM-VAE_96_paired, SeNM-VAE_960_paired
    testset_name = 'sidd'

    nls = [2, 2]
    num_down = 0
    zn_dim = 16
    
    scale = 1
    
    normalize = False
    
    temperature = 1.0
    n_max = 1280
    patch_size = 256

    model_pool = 'model_zoo'             # fixed
    results = 'results'                  # fixed
    task_current = 'trans'
    result_name = testset_name + '_' + task_current + '_' + model_name

    model_path = os.path.join(model_pool, model_name+'.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # ----------------------------------------
    # L_path, G_path, H_path
    # ----------------------------------------

    L_path = 'path/to/validation/set/noisy/'
    H_path = 'path/to/validation/set/clean/'

    G_path = os.path.join(results, result_name)
    util.mkdir(G_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(G_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # ----------------------------------------
    # load model
    # ----------------------------------------
    from SIDD.models.network_senmvae import PVAE
    model = PVAE(nls=nls, num_down=num_down, zn_dim=zn_dim)
    states = torch.load(model_path)
    model.load_state_dict(states, strict=True)
    model.eval()

    for k, v in model.named_parameters():
        v.requires_grad = False

    model = model.to(device)

    logger.info('model_name:{}'.format(model_name))
    logger.info(H_path)
    H_paths = util.get_image_paths(H_path)
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    if n_max is not None:
        H_paths = H_paths[:n_max]
        L_paths = L_paths[:n_max]

    for idx, img in enumerate(tqdm(H_paths)):

        # ------------------------------------
        # (1) img_H
        # ------------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        img_H = util.imread_uint(img, n_channels=3)
        img_L = util.imread_uint(L_paths[idx], n_channels=3)

        H, W = img_H.shape[:2]

        patch_H = img_H[H//2 - patch_size//2 : H//2 + patch_size//2, W//2 - patch_size//2 : W//2 + patch_size//2, :]
        patch_L = img_L[H//2 - patch_size//2 : H//2 + patch_size//2, W//2 - patch_size//2 : W//2 + patch_size//2, :]

        img_H = util.uint2tensor4(patch_H).to(device)
        img_L = util.uint2tensor4(patch_L).to(device)

        # ------------------------------------
        # (2) img_G
        # ------------------------------------

        input_noise_level = random_noise_levels_sidd(1).to(device) / scale
        img_H = img_H / scale
        
        img_G = model.translate(img_H, input_noise_level, temperature=temperature)

        if normalize:
            img_G = img_G - torch.mean(img_G, dim=(2, 3), keepdim=True) + torch.mean(img_H, dim=(2, 3), keepdim=True)
        
        img_G = img_G * scale
        img_H = img_H * scale

        img_G = util.tensor2uint(img_G)
        util.imsave(img_G, os.path.join(G_path, img_name+'_G'+ext))

if __name__ == '__main__':

    main()
