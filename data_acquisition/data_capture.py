import os
import time
import argparse

""" 
Run this demo by this command:
python data_capture.py -n 7 -t 1.5

open the camera with this command:
adb shell am start -a android.media.action.STILL_IMAGE_CAMERA

take photo by your camera with this command:
adb shell input keyevent 27 

temporarily quit the camera with this command:
adb shell input keyevent 4

screenshot by your phone with this command:
adb shell /system/bin/screencap -p /sdcard/screenshot.png

save the screenshot to your phone with this command:
adb pull /sdcard/screenshot.png screenshot.png

pull one image file onto your computer with this command (~/aaaa.png is the path of your computer):
adb pull /storage/emulated/0/DCIM/Camera/IMG_20210413_190300.jpg ~/aaaa.png 
"""

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_photos", type=int, help="Number of photos for one scene.")
parser.add_argument("-t", "--time_delay", type=float, help="Time delay between two photos.")
args = parser.parse_args()

# open the camera
os.system("adb shell am start -a android.media.action.STILL_IMAGE_CAMERA")
# set the iterations for taking photos and time delay for post processing.
for i in range(args.num_photos):
    time.sleep(args.time_delay)
    os.system("adb shell input keyevent 27")
    
