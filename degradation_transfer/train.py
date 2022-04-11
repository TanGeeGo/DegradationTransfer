import os
from matplotlib import image
import tqdm
import time
import torch
import cv2
import math
import argparse
import tifffile

import numpy as np

from utils import create_dir, im2double
from train_func import train_function

# CUDA_VISIBLE_DEVICES=0 python train.py -d ~/data/ -o ~/output/ --region 0.0 0.5 0.0 0.5 --white_balance 1.938645 1.000000 1.889194
# CUDA_VISIBLE_DEVICES=1 python train.py -d ~/data/ -o ~/output/ --region 0.0 0.5 0.5 1.0 --white_balance 1.938645 1.000000 1.889194
# CUDA_VISIBLE_DEVICES=2 python train.py -d ~/data/ -o ~/output/ --region 0.5 1.0 0.0 0.5 --white_balance 1.938645 1.000000 1.889194
# CUDA_VISIBLE_DEVICES=3 python train.py -d ~/data/ -o ~/output/ --region 0.5 1.0 0.5 1.0 --white_balance 1.938645 1.000000 1.889194

# training information
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--datapath", type=str, 
    help="training data path")
parser.add_argument("-o", "--outpath", type=str,
    help="output data path")
parser.add_argument("-r", "--region", nargs="+", type=float, default=[0, 0.5, 0, 0.5],
    help="training region, e.g., [h1, h2, w1, w2] is the percentage of the region to the whole image")
parser.add_argument("-s", "--imgshape", type=list, default=[3000, 4000],
    help="shape of the whole image, e.g., [h, w] is the h and w of image")
parser.add_argument("-w", "--white_balance", nargs="+", type=float, default=[1.5000, 1.0000, 1.5000],
    help="white balance of the image")
parser.add_argument("-i", "--stride", type=int, default=50,
    help="the stride for kernel prediction")
args = parser.parse_args()

# create directory for saving results
print("The estimated kernel of each iteration is saved in {:s} \n".format(args.outpath))
create_dir(args.outpath)

# the pixels of the whole Field of View
full_fov_pix = math.sqrt((args.imgshape[0]/2)**2 + (args.imgshape[1]/2)**2)
img_center = [args.imgshape[0] / 2, args.imgshape[1] / 2]
# the Field of View for kernel prediction
h_fov_range = range(args.imgshape[0]*args.region[0]+args.stride, 
                    args.imgshape[0]*args.region[1]+1, args.stride)
w_fov_range = range(args.imgshape[1]*args.region[2]+args.stride, 
                    args.imgshape[1]*args.region[3]+1, args.stride)
# begin the iteration
for h in h_fov_range:
    for w in w_fov_range:
        # output message
        print("*"*100 + "\nH: {:04d}, W: {:04d} is start\n".format(h, w) + "*"*100)
        start_time = time.time()
        # calculate the fov of this patch
        center = [(h + (h - args.stride)) / 2, (w + (w - args.stride)) / 2]
        fov_pix = math.sqrt(np.abs(img_center[0] - center[0])**2 + 
                            np.abs(img_center[1] - center[1])**2)
        fov = fov_pix / full_fov_pix
        
        # change the configuration according to the fov
        if (0.8 <= fov) and (fov <= 1):
            from option.option_krnsz27x27 import Config
        elif (0.5 <= fov) and (fov < 0.8):
            from option.option_krnsz23x23 import Config
        elif (0.0 <= fov) and (fov < 0.5):
            from option.option_krnsz19x19 import Config

        # load the input data pairs
        input_path = os.path.join(args.datapath, "input", 
                                  "h_{:04d}_w_{:04d}.tiff".format(h, w))
        label_path = os.path.join(args.datapath, "label", 
                                  "h_{:04d}_w_{:04d}.tiff".format(h, w))
        input = im2double(tifffile.imread(input_path))
        label = im2double(tifffile.imread(label_path))
        # add white balance information
        conf = Config().parse(args=["--output_dir", args.outpath,
                                    "--img_name", "h_{:04d}_w_{:04d}".format(h, w),
                                    "--v_input_dir", os.path.join(args.datapath, "valid")])
        if len(input.shape) == 2:
            # gray image, 0 means gray image
            train_function(conf, args, input, label, 0)
        elif len(input.shape) == 3:
            # split channel for kernel prediction
            for ch in range(input.shape[2]):
                # output message
                print("*"*60 + "\nNow is the {:d} channel\n".format(ch) + "*"*60)
                input_ch = input[:, :, ch]
                label_ch = label[:, :, ch]
                # begin training 
                train_function(conf, args, input_ch, label_ch, ch+1)
                # save input and label for comparisons
                cv2.imwrite(os.path.join(args.outpath,
                                         "h_{:04d}_w_{:04d}".format(h, w),
                                         "ori_img_pred",
                                         "input_{:d}.png".format(ch)),
                                         input_ch*255.)
                cv2.imwrite(os.path.join(args.outpath,
                                         "h_{:04d}_w_{:04d}".format(h, w),
                                         "ori_img_pred",
                                         "label_{:d}.png".format(ch)),
                                         label_ch*255.)
        
        # output message
        print("*"*100 + "\n\nH: {:04d}, W: {:04d} is finished\n".format(h, w) + \
              "Time consuming: {:08f}\n\n".format(time.time() - start_time) + "*"*100)