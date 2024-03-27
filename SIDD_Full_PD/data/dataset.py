import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util


class Dataset(data.Dataset):
    def __init__(self, opt):
        super(Dataset, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 128
        
        self.dataroot_H = opt['dataroot_H']
        self.dataroot_L = opt['dataroot_L']
        
        self.paths_H = util.get_image_paths(self.dataroot_H)
        self.paths_L = util.get_image_paths(self.dataroot_L)
        
        assert len(self.paths_H) == len(self.paths_L)
        
        n_max = opt['n_max']
        
        if n_max is not None:
            self.paths_H = self.paths_H[:n_max]
            self.paths_L = self.paths_L[:n_max]

    def __getitem__(self, index):
        
        path_H = self.paths_H[index]
        path_L = self.paths_L[index]
        
        img_H = util.imread_uint(path_H, self.n_channels)
        img_L = util.imread_uint(path_L, self.n_channels)
        
        H, W, _ = img_L.shape
        
        if self.opt['phase'] == 'train':
            # --------------------------------
            # randomly crop L patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            mode = np.random.randint(0, 8)
            patch_H = util.augment_img(patch_H, mode=mode)
            patch_L = util.augment_img(patch_L, mode=mode)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_H = util.uint2tensor3(patch_H)
            img_L = util.uint2tensor3(patch_L)            
        else:
            img_H = util.uint2tensor3(img_H)
            img_L = util.uint2tensor3(img_L)
            
        return {'img_H': img_H, 'img_L': img_L, 'path_H': path_H, 'path_L': path_L}

    def __len__(self):
        return len(self.paths_H)



