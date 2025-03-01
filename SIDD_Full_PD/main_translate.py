import os.path
import torch
import torch.distributions as dist
from utils import utils_image as util

from tqdm import tqdm
import argparse

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
    std = torch.sqrt(variance)
    return std.unsqueeze(2).unsqueeze(3)

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    parser = argparse.ArgumentParser(description='From clean RGB images, generate {RGB_clean, RGB_noisy} pairs')
    parser.add_argument('--model_path', type=str, help='Path of pretrained model')
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

    from .models.network_senmvae import SeNMVAE
    model = SeNMVAE(nls=nls)
    states = torch.load(model_path)
    model.load_state_dict(states, strict=True)
    model.eval()

    for k, v in model.named_parameters():
        v.requires_grad = False

    model = model.to(device)

    H_paths = util.get_image_paths(H_path)

    if n_max is not None:
        H_paths = H_paths[:n_max]
        
    num_imgs = len(H_paths)

    if num_imgs % batch_size != 0:
        key_idx = num_imgs - (num_imgs % batch_size)
    else:
        key_idx = None
    
    idx = 0
    
    for img in tqdm(H_paths):
        if idx == key_idx:
            batch_size = (num_imgs % batch_size)
            img_names = []
            img_Hs = torch.empty(batch_size, 3, img_size, img_size).to(device)
            idx = 0
        elif idx % batch_size == 0:
            img_names = []
            img_Hs = torch.empty(batch_size, 3, img_size, img_size).to(device)

        img_name, ext = os.path.splitext(os.path.basename(img))
        img_H = util.imread_uint(img, n_channels=3)

        img_H = util.uint2tensor4(img_H).to(device)

        img_names.append(img_name)
        img_Hs[idx % batch_size] = img_H[0]

        if (idx + 1) % batch_size == 0:            
            input_noise_levels = random_noise_levels_sidd(img_Hs.shape[0]).to(device)

            img_Es = model.generate(img_Hs, input_noise_levels, temperature=1.0)
                                    
            if normalize:
                img_Es = img_Es - torch.mean(img_Es, dim=(2, 3), keepdim=True) + torch.mean(img_Hs, dim=(2, 3), keepdim=True)

            for i, img_name in enumerate(img_names):
                img_E = img_Es[i][None, ...]
                img_E = util.tensor2uint(img_E)

                util.imsave(img_E, os.path.join(G_path, img_name + ext))

        idx = idx + 1

if __name__ == '__main__':

    main()