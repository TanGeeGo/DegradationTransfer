## How to generate the ideal patch from the real checker is illustrated here.

### 1. Obtain the checkerboard post-processed by the procedure in data_acquisition:

<div align=center>
<img src = "https://github.com/TanGeeGo/DegradationTransfer/blob/main/backward_transfer/data/IMG_20211229_141728.jpg">
</div>

### 2. Generate the ideal patch from the real checkerboard

* #### After modifing the parameters for backward transfer, run this demo:

```matlab
>>> patch_generator.m
```

### 3. Generate the ideal patch from the real checkerboard

* #### Pull the raw whiteboard image from camera to your laptop:


![image](https://github.com/TanGeeGo/DegradationTransfer/blob/main/backward_transfer/data/h_0300_w_0900_input.png)
![image](https://github.com/TanGeeGo/DegradationTransfer/blob/main/backward_transfer/data/h_0300_w_0900_label.png)

* #### Left is the input patch of the deep-linear network and the model is supervised by the right patch
