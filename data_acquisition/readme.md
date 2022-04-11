## How to collect the data for your own android devices is illustrated here.

### 1. prepare your camera as follows:

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
