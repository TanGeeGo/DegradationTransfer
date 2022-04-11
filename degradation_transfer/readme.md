## Training the degradation transfer of each FoV separately.

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

Note: the image path in the "data_generator.py" needs to be changed, such as the label image path, the output image path, and the PSFs path.
