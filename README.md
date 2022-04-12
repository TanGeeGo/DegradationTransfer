# DegradationTransfer & FoV-KPN (ICCV, 2021)
by Shiqi Chen, Keming Gao, Huajun Feng, Zhihai Xu, Yueting Chen

This is the official Pytorch implementation of "**Extreme-Quality Computational Imaging via Degradation Framework**" [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Extreme-Quality_Computational_Imaging_via_Degradation_Framework_ICCV_2021_paper.html)

## Prerequisites
```python
pip install -r requirements.txt
```
Please make sure your machine has a GPU, and its driver version is comply with the CUDA version! This will reduce the problems when installing the DCNv2 module later.

The Deformable ConvNets V2 (DCNv2) module in our code adopts [Xujiarui's Implementation](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0). We recommand you recompile the code according to your machine and python environment as follows:

```python
cd ~/dcn
python setup.py develop
```

This may cause many issues, please open Issues and ask me if you have any problems!
