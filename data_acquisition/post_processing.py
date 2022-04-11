import os
import cv2
import glob
import time
import scipy.io
import argparse
import tifffile
import numpy as np

"""
Post processing the dng file to 16-bit tiff file
Support the postprocessing for different camera's data

Run this demo by this command:
python post_processing.py -i ~/rawdata -n 7 -e ~/env_illu.mat -d 1.0
"""

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_path", type=str, help="The path of the real dng photo.")
parser.add_argument("-n", "--n_samples", type=int, help="N raw dng file to one image.")
parser.add_argument("-e", "--env_illu_path", type=str, help="Shading information of lens and environment.")
parser.add_argument("-d", "--downscale", type=float, help="The factor for down-scaling saving.")
args = parser.parse_args()

# img directory of all camera
cam_dict = sorted(glob.glob(args.input_path + "/camera01"))
# load the shading information of lens
shd_mat = scipy.io.loadmat(args.shading_path)
shd_mat = shd_mat["shading"]
# load the environment information 
env_mat = scipy.io.loadmat(args.env_illu_path)
env_mat = env_mat["ratio"].astype(np.float32)
# go through different camera folder
for cam in cam_dict:
    # print camera message
    print("*"*60 + "\nNow processing the {}\n".format(cam.split('/')[-1]) + "*"*60)
    # raw image in one camera folder
    img_dict = sorted(glob.glob(cam + "/*.dng"))
    # if there is no dng files 
    if img_dict == []:
        continue # skip this iterate

    # calculate image number by N raw dng file
    if len(img_dict) % args.n_samples:
        # wrong image numbers in total number
        raise ValueError("Incorrect dng file numbers. \
            Should be the multiple of {:d}".format(args.n_samples))
    else:
        img_num = len(img_dict) // args.n_samples
    
    # go through the image to be post processed
    for img in range(img_num):
        img_path = img_dict[img*args.n_samples : (img+1)*args.n_samples]
        for idx, path in enumerate(img_path):
            # post process the raw image with dcraw
            # to get the rawlike image after demosaic
            ret = os.system("dcraw -v -4 -T -w -n 300 -q 3 -o 0 " + path)
            # img path after post processing
            rawlike_path = os.path.splitext(path)[0] + ".tiff"
            # load the img after post processing
            rawlike_img = tifffile.imread(rawlike_path)
            rawlike_img = rawlike_img.astype(np.float32)
            # superposition the raw like image
            if idx == 0:
                rawlike = rawlike_img
            else:
                rawlike = rawlike + rawlike_img

        # average the rawlike image to supress the noise
        rawlike = rawlike / args.n_samples
        # correct the shading and the env illumination of lens
        rawlike[:, :, 0] = (rawlike[:, :, 0] / shd_mat[:, :, 0]) / env_mat[:, :, 0]
        rawlike[:, :, 1] = (rawlike[:, :, 1] / shd_mat[:, :, 1]) / env_mat[:, :, 1]
        rawlike[:, :, 2] = (rawlike[:, :, 2] / shd_mat[:, :, 2]) / env_mat[:, :, 1]
        rawlike = np.clip(rawlike, 0.0, 65534.0)
        # down sample the rawlike image for saving
        rawlike = cv2.resize(rawlike, 
            (int(rawlike.shape[1]*args.downscale), int(rawlike.shape[0]*args.downscale)),
            interpolation=cv2.INTER_AREA)
        rawlike = rawlike.astype(np.uint16)
        # save the raw file after shading correction
        tifffile.imwrite(os.path.splitext(img_path[0])[0] + "_out.tiff", rawlike)
        # delete the middle tiff file
        for path in img_path:
            os.remove(os.path.splitext(path)[0] + ".tiff")

        # print image message
        print("*"*60 + "\n{} is processed\n".format(img_path[0].split("/")[-1]) + "*"*60)
        
