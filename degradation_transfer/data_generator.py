import os
import cv2
import time
import math
import glob
import scipy.io
import scipy.ndimage
import tifffile
import numpy as np
from torch import zeros_like

from utils import create_dir
from functools import partial
from multiprocessing import Pool
from multiprocessing import freeze_support

def apply_cmatrix(img, ccm):
    img_cp = np.zeros_like(img)
    img_cp[:, :, 0] = ccm[0, 0] * img[:, :, 0] + ccm[0, 1] * img[:, :, 1] + ccm[0, 2] * img[:, :, 2]
    img_cp[:, :, 1] = ccm[1, 0] * img[:, :, 0] + ccm[1, 1] * img[:, :, 1] + ccm[1, 2] * img[:, :, 2]
    img_cp[:, :, 2] = ccm[2, 0] * img[:, :, 0] + ccm[2, 1] * img[:, :, 1] + ccm[2, 2] * img[:, :, 2]
    return img_cp

def unprocess(img, wb, ccm):
    wb_idx = np.random.randint(0, 4)
    wb_chosen = wb[wb_idx, :]
    # inverse ccm
    img_inv_ccm = apply_cmatrix(img, ccm)
    img_inv_ccm = np.clip(img_inv_ccm, 0, 1)
    img_inv_gamma = img_inv_ccm ** 2.2 # 1e-9
    img_inv_gamma = np.clip(img_inv_gamma, 0, 1)
    img_inv_wb = np.concatenate((np.expand_dims(img_inv_gamma[:, :, 0] / wb_chosen[0], axis=2), \
                                 np.expand_dims(img_inv_gamma[:, :, 1] / wb_chosen[1], axis=2), \
                                 np.expand_dims(img_inv_gamma[:, :, 2] / wb_chosen[2], axis=2)), axis=2)
    img_inv_wb = np.clip(img_inv_wb, 0, 1)
    return img_inv_wb, wb_chosen

def process(img, wb, ccm):
    img_wz_wb = np.concatenate((np.expand_dims(img[:, :, 0] * wb[0], axis=2), \
                                np.expand_dims(img[:, :, 1] * wb[1], axis=2), \
                                np.expand_dims(img[:, :, 2] * wb[2], axis=2)), axis=2)
    img_wz_wb = np.clip(img_wz_wb, 0, 1)
    img_wz_gamma = img_wz_wb ** (1/2.2)
    img_wz_ccm = apply_cmatrix(img_wz_gamma, ccm)
    img_wz_ccm = np.clip(img_wz_ccm, 0, 1)
    return img_wz_ccm

def gaussian(kernel_size, sigma):
    '''
    generate a gaussian kernel
    args:
        kernel_size: size of the kernel, int
        sigma: sigma of the gaussian distribution, float
    '''
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    if sigma<=0:
        sigma = ((kernel_size-1)*0.5-1)*0.3+0.8

    s = sigma ** 2
    sum_val = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i-center, j-center
            
            kernel[i, j] = np.exp(-(x**2 + y**2)/(2*s))
            sum_val += kernel[i, j]

    kernel = kernel / sum_val

    return kernel

def conv(img_path, label_dir, input_dir, kernel_path, patch_itv, patch_size, wb, ccm, ccm_inv):
    # read the image and convert BGR to RGB
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = img[..., ::-1]
    img = np.asarray(img / 255, np.float32)
    # split the filename
    filename = (img_path.split('/')[-1]).split('.')[0]
    ##
    print('Image %s is conved!' % img_path)
    ##
    # H and W of the image
    [H, W, ch] = img.shape
    """unprocess the image"""
    img_unprocessed, wb_chosen = unprocess(img, wb, ccm_inv)
    # initilize the output image
    img_degraded = np.zeros_like(img)
    # pad the image for convolution
    pad_length = int((patch_size - patch_itv) / 2)
    img_unprocessed = np.pad(img_unprocessed, ((pad_length, pad_length), (pad_length, pad_length), (0, 0)), 'reflect')
    
    """load the degradation kernel"""
    kernel = scipy.io.loadmat(kernel_path)

    """start the iteration of patch"""
    for h_index in range(1, int(H / patch_itv) + 1, 1):
        for w_index in range(1, int(W / patch_itv) + 1, 1):
            # crop the patch for convolution
            patch = img_unprocessed[(h_index-1)*patch_itv : (h_index-1)*patch_itv+patch_size, \
                                    (w_index-1)*patch_itv : (w_index-1)*patch_itv+patch_size, :]

            kernel_r = kernel['h_{:04d}_w_{:04d}_r'.format(h_index*patch_itv, w_index*patch_itv)]
            kernel_g = kernel['h_{:04d}_w_{:04d}_g'.format(h_index*patch_itv, w_index*patch_itv)]
            kernel_b = kernel['h_{:04d}_w_{:04d}_b'.format(h_index*patch_itv, w_index*patch_itv)]
            # normalize the kernel
            kernel_r = kernel_r / np.sum(kernel_r)
            kernel_g = kernel_g / np.sum(kernel_g)
            kernel_b = kernel_b / np.sum(kernel_b)

            """convolute the patch with the kernel"""
            patch_r_degraded = scipy.ndimage.convolve(patch[:, :, 0], kernel_r, mode='reflect')
            patch_g_degraded = scipy.ndimage.convolve(patch[:, :, 1], kernel_g, mode='reflect')
            patch_b_degraded = scipy.ndimage.convolve(patch[:, :, 2], kernel_b, mode='reflect')

            patch_degraded = np.concatenate((np.expand_dims(patch_r_degraded, axis=2), \
                                             np.expand_dims(patch_g_degraded, axis=2), \
                                             np.expand_dims(patch_b_degraded, axis=2)), axis=2)
            # clip the odd value
            patch_degraded = np.clip(patch_degraded, 0, 1)
            """crop the patch and paste it to the degraded image"""
            img_degraded[(h_index-1)*patch_itv : (h_index-1)*patch_itv+patch_itv, \
                         (w_index-1)*patch_itv : (w_index-1)*patch_itv+patch_itv, :] = \
                            patch_degraded[pad_length : pad_length+patch_itv, \
                                                   pad_length : pad_length+patch_itv, :]

    img_degraded_wzwb = process(img_degraded, wb_chosen, ccm)
    img_label_wzwb = process(img_unprocessed[pad_length : H+pad_length, pad_length : W+pad_length, :], wb_chosen, ccm)      
    """save the degradation results"""
    tifffile.imwrite(os.path.join(input_dir, (filename + '.tiff')), (img_degraded_wzwb * 65535).astype(np.uint16))
    print('Image is saved in path: %s' % (os.path.join(input_dir, filename)))
    tifffile.imwrite(os.path.join(label_dir, (filename + '.tiff')), (img_label_wzwb * 65535).astype(np.uint16))
    print('Image is saved in path: %s' % (os.path.join(label_dir, filename)))

if __name__ == "__main__":
    # input image path
    # label8bit_dir = '/hdd4T_2/Aberration2021/synthetic_datasets/camera03/train_datasets/label_8bit'
    label8bit_dir = '/hdd4T_2/Aberration2021/synthetic_datasets/camera03/valid_datasets/label_8bit'
    # label raw path
    # labelraw_dir = '/hdd4T_2/Aberration2021/synthetic_datasets/camera04/train_datasets/label_rgb'
    labelraw_dir = '/hdd4T_2/Aberration2021/synthetic_datasets/camera04/valid_datasets/label_rgb'
    create_dir(labelraw_dir)
    # output image path
    # inputraw_dir = '/hdd4T_2/Aberration2021/synthetic_datasets/camera04/train_datasets/input_rgb_20220325'
    inputraw_dir = '/hdd4T_2/Aberration2021/synthetic_datasets/camera04/valid_datasets/input_rgb_20220329'
    create_dir(inputraw_dir)
    # kernel path
    kernel_path = '/hdd4T_2/Aberration2021/checker/camera04/kernelpred/kernel/kernel_20220329.mat'
    # interval of the patch, 10pixels
    patch_itv = 50
    # size of the patch, 100pixels
    patch_size = 150
    # white balance of the input image
    # wb = [2.237, 1.066, 1.911]
    # generate the image list
    img_list = sorted(glob.glob(label8bit_dir + '/*.png'))

    ccm = np.array([[ 1.93994141, -0.73925781, -0.20068359],
                    [-0.28857422,  1.59741211, -0.30883789],
                    [-0.0078125 , -0.45654297,  1.46435547]])

    ccm_inv = np.array([0.560585587345286, 0.299436361600599, 0.139978051054115,\
                        0.108381485800569, 0.724058690188644, 0.167559824010787, \
                        0.036780946522526, 0.227337731614535, 0.735881321862938]).reshape(3, 3)
    
    # ccm_inv = np.linalg.inv(ccm)
    wb = np.array([1.8910, 1, 1.8031,
                   1.8031, 1, 1.7400,
                   2.0156, 1, 1.7308,
                   1.7436, 1, 1.9560]).reshape(4, 3)

    # start the multiprocessing pool
    with Pool(36) as pool:
        pool.map(partial(conv, label_dir=labelraw_dir, input_dir=inputraw_dir, kernel_path=kernel_path, \
            patch_itv=patch_itv, patch_size=patch_size, wb=wb, ccm=ccm, ccm_inv=ccm_inv), img_list)
