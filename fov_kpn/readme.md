
## This is the official Pytorch implementation of FoV-KPN.

### 1. prepare the dataset of your camera by:

```python
python dataset_generator.py
```

Note that the path information in this file needs update to the path of your computer:

```python
date_ind = "20220329" # date information for h5py file
dataset_type = "valid" # type of dataset "train" or "valid"
camera_idx = "camera04" # index of camera "camera01" to "camera05" 
base_path = "/hdd4T_2/Aberration2021/synthetic_datasets" # system path 
input_dir = "input_rgb_20220329" # input data dir
label_dir = "label_rgb" # label data dir
if_mask = False # whether add mask
# split FoV for dataset generation
# splited_fov = [0.0, 0.3, 0.6, 0.9, 1.0]
splited_fov = [0.0, 1.0]
```

### 2. Check the option file information

* #### Checking the data path and other hyper-parameters for training   

### 3. Training the FoV-KPN

```python
python train.py
```

### 4. Test on the actual photographs of your camera

* #### Pull the raw image from camera to your laptop:

```
adb pull -r ~/DCIM/Camera/*.dng ~/rawdata
```

* #### Postprocessing the captured raw images by:

```python
python post_processing.py -i ~/rawdata -n 7 -e ~/env_illu.mat -d 1.0
```

The 16-bit image is saved in the same directory of rawdata, named with "*_out.tiff"
