# DegradationTransfer & FoV-KPN (ICCV, 2021)
by Shiqi Chen, Keming Gao, Huajun Feng, Zhihai Xu, Yueting Chen

This is the official Pytorch implementation of "**Extreme-Quality Computational Imaging via Degradation Framework**" [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Extreme-Quality_Computational_Imaging_via_Degradation_Framework_ICCV_2021_paper.html)

## Prerequisites
* Python 3.7
* Matlab
* Other python packages are downloaded as follows:
```python
pip install -r requirements.txt
```
*None*: Please make sure your machine has a GPU, and its driver version is comply with the CUDA version! This will reduce the problems when installing the DCNv2 module later.

The Deformable ConvNets V2 (DCNv2) module in our code adopts [Xujiarui's Implementation](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0). We recommand you recompile the code according to your machine and python environment as follows:

```python
cd ~/dcn
python setup.py develop
```

This may cause many issues, please open Issues and ask me if you have any problems!

## Data Acquisition

### 1. prepare your camera and experimental devices as follows:

<div align=center>
<img src = "https://github.com/TanGeeGo/DegradationTransfer/blob/main/data_acquisition/explanatory_material/experimental_devices.png">
</div>

### 2. Calibrate the environment illuminance

* #### Download the [Android Debug Bridge (ADB)](https://source.android.com/setup/build/adb) to your laplop, and link your camera with your laptop.

* #### Please ensure ADB is in the system parameters of your laptop.

* #### Shooting the pure white scene by:

```python
python data_capture.py -n 1 -t 1.5
```

* #### Pull the raw whiteboard image from camera to your laptop:

```
adb pull ~/DCIM/Camera/whiteboard.dng ~/whiteboard
```

* #### Download the [dcraw](https://www.dechifro.org/dcraw/) and ensure dcraw is in the system parameters.

* #### Postprocessing the captured whiteboard raw image with [dcraw](https://www.dechifro.org/dcraw/):

```
dcraw -v -4 -T -w -n 300 -q 3 -o 0 ~/whiteboard/whiteboard.dng
```

* #### Calibrate the environment illuminance:

```python
python env_illuminance.py -i ~/whiteboard/whiteboard.tiff -o ~/env_illu.mat -p 100
```

### 3. Checkerboard capture and postprocessing

* #### Capture the checkerboard by:

```python
python data_capture.py -n 7 -t 1.5
```

* #### Pull the raw image from camera to your laptop:

```
adb pull -r ~/DCIM/Camera/*.dng ~/rawdata
```

* #### Postprocessing the captured raw images by:

```python
python post_processing.py -i ~/rawdata -n 7 -e ~/env_illu.mat -d 1.0
```

The 16-bit image is saved in the same directory of rawdata, named with "*_out.tiff"

## Backward Transfer

### 1. Obtain the checkerboard post-processed by the procedure in data acquisition:

<div align=center>
<img src = "https://github.com/TanGeeGo/DegradationTransfer/blob/main/backward_transfer/data/IMG_20211229_141728.jpg">
</div>

### 2. Generate the ideal patch from the real checkerboard

* #### After modifing the parameters for backward transfer, run this demo in matlab:

```matlab
>>> patch_generator.m
```

### 3. Generate the ideal patch from the real checkerboard

* #### Pull the raw whiteboard image from camera to your laptop:


![image](https://github.com/TanGeeGo/DegradationTransfer/blob/main/backward_transfer/data/h_0300_w_0900_input.png)
![image](https://github.com/TanGeeGo/DegradationTransfer/blob/main/backward_transfer/data/h_0300_w_0900_label.png)

* #### Left is the input patch of the deep-linear network and the model is supervised by the right patch

## Degradation Transfer

### 1. After prepare the paired patches in "../backward_transfer/data/input" and "../backward_transfer/data/label", the training can be performed by:

```python
# CUDA_VISIBLE_DEVICES=0 python train.py -d ../backward_transfer/data/ -o ~/output/ --region 0.0 0.5 0.0 0.5 --white_balance 1.938645 1.000000 1.889194
# CUDA_VISIBLE_DEVICES=1 python train.py -d ../backward_transfer/data/ -o ~/output/ --region 0.0 0.5 0.5 1.0 --white_balance 1.938645 1.000000 1.889194
# CUDA_VISIBLE_DEVICES=2 python train.py -d ../backward_transfer/data/ -o ~/output/ --region 0.5 1.0 0.0 0.5 --white_balance 1.938645 1.000000 1.889194
# CUDA_VISIBLE_DEVICES=3 python train.py -d ../backward_transfer/data/ -o ~/output/ --region 0.5 1.0 0.5 1.0 --white_balance 1.938645 1.000000 1.889194
```

* #### The options for different FoV can be modified, we recommand you change the option according to the performance of this FoV.

* #### The training split the FoV into 4 regions and the white balance of this checkerboard is needed for unprocess.

* #### We recommend you separately train the model for higher efficiency.

### 2. The predicted PSFs are saved by training, collect the test PSFs of each FoV.

```python
python kernel_sort.py -d ~/ -o ~/kernel/
```

### 3. Use the PSFs of different FoVs to generate data pairs.

```python
python data_generator.py
```

Note that the image path in the "data_generator.py" needs to be changed, such as the label image path, the output image path, and the PSFs path:

```python
# input image path
# label8bit_dir = '~/train_datasets/label_8bit'
label8bit_dir = '~/valid_datasets/label_8bit'
# label raw path
# labelraw_dir = '~/train_datasets/label_rgb'
labelraw_dir = '~/valid_datasets/label_rgb'
create_dir(labelraw_dir)
# output image path
# inputraw_dir = '~/train_datasets/input_rgb'
inputraw_dir = '~/valid_datasets/input_rgb'
create_dir(inputraw_dir)
# kernel path
kernel_path = '~/kernel/kernel.mat'
```

## FoV-KPN

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

Note: The training information and the test information are in the same option.py file!

### 3. Training the FoV-KPN

```python
python train.py
```

### 4. Test on the actual photographs of your camera

```python
python test_real.py
```
