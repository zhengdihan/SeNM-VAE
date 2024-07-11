import numpy as np
from utils import utils_image as util
from collections import OrderedDict
import os.path
from scipy.stats import entropy

def compute_kl(img1, img2):
    hist1, _ = np.histogram(img1.ravel(), bins=256, range=(0, 255))
    hist2, _ = np.histogram(img2.ravel(), bins=256, range=(0, 255))

    # Compute the probability distributions of pixel values for each image
    prob1 = hist1 / np.sum(hist1)
    prob2 = hist2 / np.sum(hist2)
    prob2 = (prob2 + 1e-8)
    prob2 = prob2 / np.sum(prob2)

    # Compute the KL divergence between the two probability distributions
    kl_divergence = entropy(prob1, qk=prob2)

    return kl_divergence

def main():
    G_path = '' # Data path for generated noisy images
    R_path = '' # Data path for real noisy images

    test_results = OrderedDict()
    test_results['kl'] = []

    G_paths = util.get_image_paths(G_path)
    R_paths = util.get_image_paths(R_path)

    for idx, img in enumerate(G_paths):

        img_name, ext = os.path.splitext(os.path.basename(img))

        img_G = util.imread_uint(img, n_channels=1)
        img_R = util.imread_uint(R_paths[idx], n_channels=1)

        # --------------------------------
        # PSNR and SSIM
        # --------------------------------

        kl = compute_kl(img_G, img_R)
        test_results['kl'].append(kl)
        print('{:s} - KL: {:.4f}.'.format(img_name+ext, kl))

    ave_kl = sum(test_results['kl']) / len(test_results['kl'])

    print('Average KL: {:.6f}.'.format(ave_kl))

if __name__ == '__main__':
    main()
