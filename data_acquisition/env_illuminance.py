import os
import time
import tifffile
import scipy.io
import scipy.interpolate
import numpy as np

"""
    environment illuminance calibration
"""
# load the white board image after post processing 
image_path = "~/whiteboard.tiff"
image = tifffile.imread(image_path)
# calculate averange pixel value per channel
mean_r = np.mean(image[:, :, 0])
mean_g = np.mean(image[:, :, 1])
mean_b = np.mean(image[:, :, 2])

# seprate the image into 100 x 100 patches
patch_size = 100
m, n = image.shape[0], image.shape[1]
h_range = np.arange(patch_size//2, m, patch_size)
w_range = np.arange(patch_size//2, n, patch_size)
illu_ratio = np.zeros((len(h_range), len(w_range), 3))
for h_idx, h in enumerate(h_range):
    for w_idx, w in enumerate(w_range):
        # crop the patch and evaluate the mean
        patch = image[(h-patch_size//2) : (h+patch_size//2),
                      (w-patch_size//2) : (w+patch_size//2), :]
        illu_ratio[h_idx, w_idx, 0] = np.mean(patch[:, :, 0])/mean_r
        illu_ratio[h_idx, w_idx, 1] = np.mean(patch[:, :, 1])/mean_g
        illu_ratio[h_idx, w_idx, 2] = np.mean(patch[:, :, 2])/mean_b

# interpolate the 2D illu ratio
f_illu_ratio_r = scipy.interpolate.interp2d(w_range, h_range, 
                                            illu_ratio[:, :, 0], 
                                            kind='cubic')
f_illu_ratio_g = scipy.interpolate.interp2d(w_range, h_range, 
                                            illu_ratio[:, :, 1], 
                                            kind='cubic')
f_illu_ratio_b = scipy.interpolate.interp2d(w_range, h_range, 
                                            illu_ratio[:, :, 2], 
                                            kind='cubic')
h_pixel_range = np.arange(0, m)
w_pixel_range = np.arange(0, n)
ratio_r = f_illu_ratio_r(w_pixel_range, h_pixel_range)
ratio_g = f_illu_ratio_g(w_pixel_range, h_pixel_range)
ratio_b = f_illu_ratio_b(w_pixel_range, h_pixel_range)
ratio = np.stack((ratio_r, ratio_g, ratio_b), axis=2)

# save the ratio in mat file
scipy.io.savemat('~/env_illu.mat', {"ratio": ratio})
