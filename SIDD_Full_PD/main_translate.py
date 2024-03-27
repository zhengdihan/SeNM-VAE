import numpy as np
import os.path

import torch
import torch.distributions as dist
from utils import utils_image as util

from tqdm import tqdm
import argparse

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    parser = argparse.ArgumentParser(description='From clean RGB images, generate {RGB_clean, RGB_noisy} pairs')
    parser.add_argument('--model_path', type=str, help='Path of pretrained model')
    parser.add_argument('--noise_level_path', type=str, help='Path of noise level')
    parser.add_argument('--H_path', type=str, help='Directory of clean RGB images')
    parser.add_argument('--G_path', type=str, help='Directory for synthetic noisy RGB images')
    parser.add_argument('--batch_size', type=int, default=10, help='Input batch size')
    parser.add_argument('--nls', nargs='+', type=int, default=[4, 4])
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--normalize', action='store_true')
    
    args = parser.parse_args()
    print(args)

    n_max = None
    
    model_path = args.model_path
    noise_level_path = args.noise_level_path
    H_path = args.H_path
    G_path = args.G_path
    batch_size = args.batch_size
    img_size = args.img_size
    nls = args.nls
    normalize = args.normalize
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    
    util.mkdir(G_path)

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from SIDD_Full_PD.models.network_senmvae import MVAE
    model = MVAE(nls=nls)
    states = torch.load(model_path)
    model.load_state_dict(states, strict=True)
    model.eval()

    for k, v in model.named_parameters():
        v.requires_grad = False

    model = model.to(device)

    H_paths = util.get_image_paths(H_path)

    noise_levels = np.load(noise_level_path).astype(np.float32)

    if n_max is not None:
        H_paths = H_paths[:n_max]
        noise_levels = noise_levels[:n_max]
        
    num_imgs = len(H_paths)

    if num_imgs % batch_size != 0:
        key_idx = num_imgs - (num_imgs % batch_size)
    else:
        key_idx = None
    
    idx = 0
    
    for i in tqdm(range(len(H_paths))):

        if idx == key_idx:
            batch_size = (num_imgs % batch_size)
            img_names = []
            input_noise_levels = []
            img_Ls = torch.empty(batch_size, 3, img_size, img_size).to(device)
            idx = 0
        elif idx % batch_size == 0:
            img_names = []
            input_noise_levels = []
            img_Ls = torch.empty(batch_size, 3, img_size, img_size).to(device)

        img_name, ext = os.path.splitext(os.path.basename(H_paths[i]))
        img_L = util.imread_uint(H_paths[i], n_channels=3)
        img_L = util.uint2tensor4(img_L).to(device)

        input_noise_levels.append(noise_levels[i].item())
        
        img_names.append(img_name)
        img_Ls[idx % batch_size] = img_L[0]

        if (idx + 1) % batch_size == 0:
            input_noise_levels = np.array(input_noise_levels).astype(np.float32)
            input_noise_levels = torch.from_numpy(input_noise_levels).view(img_Ls.shape[0], 1, 1, 1).to(device)
            img_Es = model.generate(img_Ls, input_noise_levels, temperature=1.0)
                        
            if normalize:
                img_Es = img_Es - torch.mean(img_Es, dim=(2, 3), keepdim=True) + torch.mean(img_Ls, dim=(2, 3), keepdim=True)

            for i, img_name in enumerate(img_names):
                img_E = img_Es[i][None, ...]
                img_E = util.tensor2uint(img_E)

                util.imsave(img_E, os.path.join(G_path, img_name + ext))

        idx = idx + 1

if __name__ == '__main__':

    main()
