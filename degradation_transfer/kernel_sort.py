import os
import time
import math
import argparse

import numpy as np
import scipy.io as sio
import scipy.ndimage as snd

from utils import create_dir

# python kernel_sort.py -d /hdd4T_2/Aberration2021/checker/camera02/kernelpred -o /hdd4T_2/Aberration2021/checker/camera02/kernelpred/kernel/
# python kernel_sort.py -d /hdd4T_2/Aberration2021/checker/camera03/kernelpred -o /hdd4T_2/Aberration2021/checker/camera03/kernelpred/kernel/

# config information
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--datapath", type=str, 
    help="training data path")
parser.add_argument("-o", "--outpath", type=str,
    help="output data path")
parser.add_argument("-r", "--region", nargs="+", type=float, default=[0, 1.0, 0, 1.0],
    help="training region, e.g., [h1, h2, w1, w2] is the percentage of the region to the whole image")
parser.add_argument("-s", "--imgshape", type=list, default=[3000, 4000],
    help="shape of the whole image, e.g., [h, w] is the h and w of image")
parser.add_argument("-i", "--stride", type=int, default=50,
    help="the stride for kernel prediction")
parser.add_argument("-k", "--kernelsize", type=int, default=27,
    help="pad the kernel into the same size")
args = parser.parse_args()

# create directory for saving results
print("The estimated kernel of is saved in {:s} \n".format(args.outpath))
create_dir(args.outpath)

# the pixels of the whole Field of View
full_fov_pix = math.sqrt((args.imgshape[0]/2)**2 + (args.imgshape[1]/2)**2)
img_center = [args.imgshape[0] / 2, args.imgshape[1] / 2]

# the Field of View for kernel prediction
h_fov_range = range(round(args.imgshape[0]*args.region[0]+args.stride), 
                    int(args.imgshape[0]*args.region[1]+1), args.stride)
w_fov_range = range(round(args.imgshape[1]*args.region[2]+args.stride), 
                    int(args.imgshape[1]*args.region[3]+1), args.stride)

kernel_dict = {}
# begin the iteration go through the kernel dir
for h in h_fov_range:
    for w in w_fov_range:
        kernel_path = os.path.join(args.datapath, 'output', 'h_{:04d}_w_{:04d}'.format(h, w), 'kernel_pred')
        # check the existance of the kernels
        kernel_existence = os.path.exists(os.path.join(kernel_path, '09999_kernel_r.mat')) and \
                           os.path.exists(os.path.join(kernel_path, '09999_kernel_g.mat')) and \
                           os.path.exists(os.path.join(kernel_path, '09999_kernel_b.mat'))
        if not kernel_existence:
            raise ValueError('Path or Kernel do not exist in h: {:04d} and w: {:04d}'.format(h, w))

        # calculate the fov of this patch
        center = [(h + (h - args.stride)) / 2, (w + (w - args.stride)) / 2]
        fov_pix = math.sqrt(np.abs(img_center[0] - center[0])**2 + 
                            np.abs(img_center[1] - center[1])**2)
        fov = fov_pix / full_fov_pix

        # load the kernel and save them into dicts
        kernel_r = sio.loadmat(os.path.join(kernel_path, '09999_kernel_r.mat'))['Kernel']
        kernel_g = sio.loadmat(os.path.join(kernel_path, '09999_kernel_g.mat'))['Kernel']
        kernel_b = sio.loadmat(os.path.join(kernel_path, '09999_kernel_b.mat'))['Kernel']

        # check the sum of kernel
        kernel_r = kernel_r / np.sum(kernel_r)
        kernel_g = kernel_g / np.sum(kernel_g)
        kernel_b = kernel_b / np.sum(kernel_b)
        
        # pad the kernel into the same size
        pad_size = (args.kernelsize-kernel_r.shape[0]) // 2
        kernel_r = np.pad(kernel_r, (pad_size, pad_size), 'constant')
        kernel_g = np.pad(kernel_g, (pad_size, pad_size), 'constant')
        kernel_b = np.pad(kernel_b, (pad_size, pad_size), 'constant')
        
        # add it to kernel directory
        kernel_dict['h_{:04d}_w_{:04d}_r'.format(h, w)] = kernel_r
        kernel_dict['h_{:04d}_w_{:04d}_g'.format(h, w)] = kernel_g
        kernel_dict['h_{:04d}_w_{:04d}_b'.format(h, w)] = kernel_b

# save the directory 
sio.savemat(os.path.join(args.outpath, 'kernel.mat'), kernel_dict)