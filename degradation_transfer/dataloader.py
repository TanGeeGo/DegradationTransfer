import os
import cv2
import random
import torch
import numpy as np
import scipy.io as sio
import matplotlib.image as mpimg

from torch.utils.data import Dataset
from utils import unprocess, process

class DataGenerator(Dataset):
    def __init__(self, conf, gan, input, label):
        super().__init__()
        # Default shapes
        self.g_input_shape = conf.input_crop_size
        self.d_input_shape = gan.G.output_size  # shape entering D downscaled by G
        
        # data pairs
        self.g_input_image = input
        self.d_input_image = label
        self.g_input_image = np.expand_dims(self.g_input_image, -1)
        self.d_input_image = np.expand_dims(self.d_input_image, -1)
        self.in_rows, self.in_cols = self.g_input_image.shape[0:2]
        # concate it for better cropping
        self.concated_input_image = np.concatenate((self.g_input_image, self.d_input_image), axis=2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        ''' Get the crop for both G and D
            the size of g_input and the size of d_input should be defined by the
            generator and the discriminator
        '''

        g_input = self.next_crop(for_g=True, idx=idx)
        d_input = self.next_crop(for_g=False, idx=idx)

        return g_input, d_input

    def next_crop(self, for_g, idx):
        """Return a crop according to the pre-determined list of indices."""
        img = self.g_input_image if for_g else self.d_input_image
        size = self.g_input_shape if for_g else self.d_input_shape
        top, left = self.get_top_left_point(for_g, idx)
        img_cropped = img[top : top+size, left : left+size]
        img_cropped = np.expand_dims(img_cropped, axis=3)
        array = np.ascontiguousarray(np.transpose(img_cropped, (3, 2, 0, 1)))
        return array

    def get_top_left_point(self, for_g, idx):
        """Translate the center of the index of the crop to it's corresponding top-left"""
        idx = idx % ((self.in_rows-self.g_input_shape+1) * (self.in_rows-self.g_input_shape+1))
        row, col = int(idx / (self.in_rows-self.g_input_shape+1)), idx % (self.in_rows-self.g_input_shape+1)
        if not for_g:
            row += int((self.g_input_shape - self.d_input_shape) / 2)
            col += int((self.g_input_shape - self.d_input_shape) / 2)
        return row, col

class DataGenerator_img(Dataset):
    def __init__(self, conf, gan):
        # Default shapes
        self.g_input_shape = conf.input_crop_size
        self.d_input_shape = gan.G.output_size  # shape entering D downscaled by G
        # self.d_output_shape = self.d_input_shape - gan.D.forward_shave

        # read input image
        self.g_input_image = mpimg.imread(os.path.join(conf.g_input_dir, conf.img_name))
        self.d_input_image = mpimg.imread(os.path.join(conf.d_input_dir, conf.img_name))
        self.g_input_image = np.expand_dims(self.g_input_image, -1)
        self.d_input_image = np.expand_dims(self.d_input_image, -1)
        self.in_rows, self.in_cols = self.g_input_image.shape[0:2]
        color_ext = conf.img_name.split('_')[-1]
        self.color_idx = color_ext.split('.')[0]
        # concate it for better cropping
        self.concated_input_image = np.concatenate((self.g_input_image, self.d_input_image), axis=2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        ''' Get the crop for both G and D
            the size of g_input and the size of d_input should be defined by the
            generator and the discriminator
        '''

        g_input = self.next_crop(for_g=True, idx=idx)
        d_input = self.next_crop(for_g=False, idx=idx)

        return g_input, d_input

    def next_crop(self, for_g, idx):
        """Return a crop according to the pre-determined list of indices."""
        img = self.g_input_image if for_g else self.d_input_image
        size = self.g_input_shape if for_g else self.d_input_shape
        top, left = self.get_top_left_point(for_g, idx)
        img_cropped = img[top : top+size, left : left+size]
        img_cropped = np.expand_dims(img_cropped, axis=3)
        array = np.ascontiguousarray(np.transpose(img_cropped, (3, 2, 0, 1)))
        return array

    def get_top_left_point(self, for_g, idx):
        """Translate the center of the index of the crop to it's corresponding top-left"""
        idx = idx % ((self.in_rows-self.g_input_shape+1) * (self.in_rows-self.g_input_shape+1))
        row, col = int(idx / (self.in_rows-self.g_input_shape+1)), idx % (self.in_rows-self.g_input_shape+1)
        if not for_g:
            row += int((self.g_input_shape - self.d_input_shape) / 2)
            col += int((self.g_input_shape - self.d_input_shape) / 2)
        return row, col

class DataGenerator_mat(Dataset):
    def __init__(self, conf, gan):
        # Default shapes
        self.g_input_shape = conf.input_crop_size
        self.d_input_shape = gan.G.output_size  # shape entering D downscaled by G
        # self.d_output_shape = self.d_input_shape - gan.D.forward_shave

        # read input mat
        g_input_image_dict = sio.loadmat(os.path.join(conf.g_input_dir, conf.img_name))
        d_input_image_dict = sio.loadmat(os.path.join(conf.d_input_dir, conf.img_name))
        self.g_input_image = g_input_image_dict['input']
        self.d_input_image = d_input_image_dict['label']
        self.g_input_image = np.expand_dims(np.expand_dims(self.g_input_image, -1), -1)
        self.d_input_image = np.expand_dims(np.expand_dims(self.d_input_image, -1), -1)
        self.g_input_image = np.transpose(self.g_input_image, (3, 2, 0, 1))
        self.d_input_image = np.transpose(self.d_input_image, (3, 2, 0, 1))
        
        self.in_rows, self.in_cols = self.g_input_image.shape[2:4]
        color_ext = conf.img_name.split('_')[-1]
        self.color_idx = color_ext.split('.')[0]
        # concate it for better cropping
        # self.concated_input_image = np.concatenate((self.g_input_image, self.d_input_image), axis=2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        ''' Get the crop for both G and D
            the size of g_input and the size of d_input should be defined by the
            generator and the discriminator
        '''

        g_input = self.next_crop(for_g=True, idx=idx)
        d_input = self.next_crop(for_g=False, idx=idx)

        return g_input, d_input

    def next_crop(self, for_g, idx):
        """Return a crop according to the pre-determined list of indices."""
        img = self.g_input_image if for_g else self.d_input_image
        size = self.g_input_shape if for_g else self.d_input_shape
        top, left = self.get_top_left_point(for_g, idx)
        img_cropped = img[:, :, top : top+size, left : left+size]
        array = np.ascontiguousarray(img_cropped)
        return array

    def get_top_left_point(self, for_g, idx):
        """Translate the center of the index of the crop to it's corresponding top-left"""
        idx = idx % ((self.in_rows-self.g_input_shape+1) * (self.in_rows-self.g_input_shape+1))
        row, col = int(idx / (self.in_rows-self.g_input_shape+1)), idx % (self.in_rows-self.g_input_shape+1)
        if not for_g:
            row += int((self.g_input_shape - self.d_input_shape) / 2)
            col += int((self.g_input_shape - self.d_input_shape) / 2)
        return row, col