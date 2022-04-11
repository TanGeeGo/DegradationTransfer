import os
import re
import torch
import numpy as np
import scipy.io as sio

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def create_penalty_mask(k_size, penalty_scale):
    """Generate a mask of weights penalizing values close to the boundaries"""
    center_size = k_size // 2 + k_size % 2
    mask = create_gaussian(size=k_size, sigma1=k_size, is_tensor=False)
    mask = 1 - mask / np.max(mask)
    margin = (k_size - center_size) // 2 - 1
    mask[margin:-margin, margin:-margin] = 0
    return penalty_scale * mask

def map2tensor(gray_map):
    """Move gray maps to GPU, no normalization is done"""
    return torch.FloatTensor(gray_map).unsqueeze(0).unsqueeze(0).cuda()

def save_final_kernel(k_2, conf):
    """saves the final kernel and the analytic kernel to the results folder"""
    sio.savemat(os.path.join(conf.output_dir_path, '%s_kernel_x2.mat' % conf.img_name), {'Kernel': k_2})
    # if conf.X4:
    #     k_4 = analytic_kernel(k_2)
    #     sio.savemat(os.path.join(conf.output_dir_path, '%s_kernel_x4.mat' % conf.img_name), {'Kernel': k_4})

def create_gaussian(size, sigma1, sigma2=-1, is_tensor=False):
    """Return a Gaussian"""
    func1 = [np.exp(-z ** 2 / (2 * sigma1 ** 2)) / np.sqrt(2 * np.pi * sigma1 ** 2) for z in range(-size // 2 + 1, size // 2 + 1)]
    func2 = func1 if sigma2 == -1 else [np.exp(-z ** 2 / (2 * sigma2 ** 2)) / np.sqrt(2 * np.pi * sigma2 ** 2) for z in range(-size // 2 + 1, size // 2 + 1)]
    return torch.FloatTensor(np.outer(func1, func2)).cuda() if is_tensor else np.outer(func1, func2)

def unprocess(img, wb, color_idx):
    # img_inv_gamma = torch.clamp(img, min=1e-8) ** 2.2
    if color_idx == 'r':
        img_inv_wb = img / wb[0]
    elif color_idx == 'g':
        img_inv_wb = img / wb[1]
    elif color_idx == 'b':
        img_inv_wb = img / wb[2]

    return img_inv_wb

def process(img, wb, color_idx):
    if color_idx == 'r':
        img = img * wb[0]
    elif color_idx == 'g':
        img = img * wb[1]
    elif color_idx == 'b':
        img = img * wb[2]

    # img_wz_gamma = torch.clamp(img_wz_wb, min=1e-8) ** (1/2.2)
    return img

def im2double(img):
    if img.dtype == "uint8":
        img = img.astype(np.float32)/255.0
    elif img.dtype == "uint16":
        img = img.astype(np.float32)/65535.0
    return img