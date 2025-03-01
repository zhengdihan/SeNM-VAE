import os.path
import torch
from utils import utils_image as util
from tqdm import tqdm
import argparse

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    parser = argparse.ArgumentParser(description='From clean RGB images, generate {RGB_clean, RGB_noisy} pairs')
    parser.add_argument('--model_path', type=str, help='Path of pretrained model')
    parser.add_argument('--H_path', type=str, help='Directory of clean RGB images')
    parser.add_argument('--L_path', type=str, help='Directory for real noisy RGB images')
    parser.add_argument('--G_path', type=str, help='Directory for synthetic noisy RGB images')
    parser.add_argument('--nls', nargs='+', type=int, default=[4, 4])
    parser.add_argument('--normalize', action='store_true')
    
    args = parser.parse_args()
    print(args)

    model_path = args.model_path
    H_path = args.H_path
    L_path = args.L_path
    G_path = args.G_path
    nls = args.nls    
    normalize = args.normalize
    temperature = 1.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # ----------------------------------------
    # G_path
    # ----------------------------------------

    util.mkdir(G_path)

    # ----------------------------------------
    # load model
    # ----------------------------------------
    from .models.network_senmvae import SeNMVAE
    model = SeNMVAE(nls=nls)
    states = torch.load(model_path)
    model.load_state_dict(states, strict=True)
    model.eval()

    for k, v in model.named_parameters():
        v.requires_grad = False

    model = model.to(device)

    H_paths = util.get_image_paths(H_path)
    L_paths = util.get_image_paths(L_path)

    for idx, img in enumerate(tqdm(H_paths)):

        # ------------------------------------
        # (1) img_H
        # ------------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        img_H = util.imread_uint(img, n_channels=3)
        img_L = util.imread_uint(L_paths[idx], n_channels=3)
        
        img_H = util.uint2tensor4(img_H).to(device)
        img_L = util.uint2tensor4(img_L).to(device)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------
        
        input_noise_level = torch.std(img_L - img_H).item()
        img_G = model.generate(img_H, input_noise_level, temperature=temperature)
        
        if normalize:
            img_G = img_G - torch.mean(img_G, dim=(2, 3), keepdim=True) + torch.mean(img_H, dim=(2, 3), keepdim=True)
            
        img_G = util.tensor2uint(img_G)
        
        # ------------------------------------
        # save results
        # ------------------------------------

        util.imsave(img_G, os.path.join(G_path, img_name+'_G'+ext))

if __name__ == '__main__':

    main()
