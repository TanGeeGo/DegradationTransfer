import h5py
import cv2
import os
import glob
import time

import numpy as np

from utils import create_dir

def crop_patch(img, half_patch_size, stride, random_crop):
    """
    crop image into patches
    input args:
        img: input image array, np.array
        half_patch_size: half of patch size, int
        stride: stride of neighbor patch, int
        random_crop: if random crop the input image, bool
    """
    patch_list = []
    [h, w, c] = img.shape
    ######################################################################################
    # calculate the fov information
    h_range = np.arange(0, h, 1)
    w_range = np.arange(0, w, 1)
    img_fld_w, img_fld_h = np.meshgrid(w_range, h_range)
    img_fld_h = ((img_fld_h - (h-1)/2) / ((h-1)/2)).astype(np.float32)
    img_fld_w = ((img_fld_w - (w-1)/2) / ((w-1)/2)).astype(np.float32)
    img_fld_h = np.expand_dims(img_fld_h, -1)
    img_fld_w = np.expand_dims(img_fld_w, -1)
    img_wz_fld = np.concatenate([img, img_fld_h, img_fld_w], 2)
    ######################################################################################
    if random_crop:
        crop_num = 100
        pos = [(np.random.randint(half_patch_size, h - half_patch_size), \
            np.random.randint(half_patch_size, w - half_patch_size)) \
            for i in range(crop_num)]
    else:
        pos = [(ht, wt) for ht in range(half_patch_size, h, stride) \
            for wt in range(half_patch_size, w, stride)]

    for (ht, wt) in pos:
        cropped_img = img_wz_fld[ht - half_patch_size:ht + half_patch_size, wt - half_patch_size:wt + half_patch_size, :]
        patch_list.append(cropped_img)

    return patch_list

def gen_dataset(src_input_files, src_label_files, dst_path):
    """
    generating datasets:
    input args: 
        src_input_files: input image files list, list[]
        src_label_files: label image files list, list[]
        dst_path: path for saving h5py file, str
    """
    h5py_path = dst_path + "/dataset.h5"
    h5f = h5py.File(h5py_path, 'w')

    for img_idx in range(len(src_input_files)):
        print("Now processing img pairs of %s", os.path.basename(src_input_files[img_idx]))
        img_input = cv2.imread(src_input_files[img_idx], cv2.IMREAD_COLOR)
        img_label = cv2.imread(src_label_files[img_idx], cv2.IMREAD_COLOR)
        img_input = img_input[..., ::-1]
        img_label = img_label[..., ::-1]

        # normalize the input and the label
        img_input = np.asarray(img_input / 255, np.float32)
        img_label = np.asarray(img_label / 255, np.float32)

        # concate input and label together
        img_pair = np.concatenate([img_input, img_label], 2)

        # crop the patch 
        patch_list = crop_patch(img_pair, 100, 100, False)

        # save the patches into h5py file
        for patch_idx in range(len(patch_list)):
            data = patch_list[patch_idx].copy()
            h5f.create_dataset(str(img_idx)+'_'+str(patch_idx), shape=(200,200,8), data=data)

    h5f.close()


if __name__ == "__main__":
    # generating train/valid/test datasets
    # dataset_type = 'train'

    src_input_path = ".../input"
    src_label_path = ".../label"
    dst_path = ".../h5py_file"
    create_dir(dst_path)

    src_input_files = sorted(glob.glob(src_input_path + "/*.png"))
    src_label_files = sorted(glob.glob(src_label_path + "/*.png"))

    print("start dataset generation!")
    gen_dataset(src_input_files, src_label_files, dst_path)