import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
import torch


class SemiSupDataset(data.Dataset):
    def __init__(self, opt):
        super(SemiSupDataset, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = self.opt['H_size'] if opt['H_size'] else 128
        
        self.labels = []
        
        self.dataroots_pair = opt['dataroots_pair']
        self.dataroots_unpair = opt['dataroots_unpair']
        
        self.dataroot_H = []
        self.dataroot_L = []
        
        self.paths_H = util.get_image_paths(self.dataroots_pair[0])
        self.paths_L = util.get_image_paths(self.dataroots_pair[1])
        
        assert len(self.paths_H) == len(self.paths_L)
        
        # -1 for paired domain
        self.labels += [-1] * len(self.paths_H)
        
        if self.dataroots_unpair is not None:
            # 0 for clean domain
            paths_cd = util.get_image_paths(self.dataroots_unpair[0])
            self.paths_H += paths_cd
            self.paths_L += paths_cd # copy
            
            self.labels += [0] * len(paths_cd)
            
            # 1 for noisy domain
            paths_nd = util.get_image_paths(self.dataroots_unpair[1])
            self.paths_H += paths_nd
            self.paths_L += paths_nd # copy
            
            self.labels += [1] * len(paths_nd)
        
        n_max = opt['n_max']
        
        if n_max is not None:
            self.paths_H = self.paths_H[:n_max]
            self.paths_L = self.paths_L[:n_max]
            self.labels = self.labels[:n_max]
        
        assert len(self.paths_H) == len(self.paths_L)
        

    def __getitem__(self, index):
        
        path_H = self.paths_H[index]
        path_L = self.paths_L[index]
        
        label = float(self.labels[index])
        label = torch.full([1, 1, 1], label).long()        
        
        img_H = util.imread_uint(path_H, self.n_channels)
        img_L = util.imread_uint(path_L, self.n_channels)
        
        H, W = img_H.shape[:2]
        
        if self.opt['phase'] == 'train':
            # ---------------------------------
            # randomly crop the patch
            # ---------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            
            # ---------------------------------
            # augmentation - flip, rotate
            # ---------------------------------
            mode = np.random.randint(0, 8)
            patch_H = util.augment_img(patch_H, mode=mode)
            patch_L = util.augment_img(patch_L, mode=mode)

            # ---------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # ---------------------------------
            img_H = util.uint2tensor3(patch_H)
            img_L = util.uint2tensor3(patch_L)
            
        else:
            patch_H = img_H[H//2 - self.patch_size//2 : H//2 + self.patch_size//2, W//2 - self.patch_size//2 : W//2 + self.patch_size//2, :]
            patch_L = img_L[H//2 - self.patch_size//2 : H//2 + self.patch_size//2, W//2 - self.patch_size//2 : W//2 + self.patch_size//2, :]
            img_H = util.uint2tensor3(patch_H)
            img_L = util.uint2tensor3(patch_L)
            
        return {'img_H': img_H, 'img_L': img_L, 'label': label, 'path_H': path_H, 'path_L': path_L}

    def __len__(self):
        return len(self.paths_H)

