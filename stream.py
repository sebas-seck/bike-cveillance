#!/usr/bin/python3
# -*- encoding: utf-8 -*-
#
# https://picamera.readthedocs.io/en/release-1.13/recipes1.html

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
from datetime import datetime
import numpy as np

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (736, 480)
camera.hflip = True
camera.vflip = True
# camera.led = False
# rawCapture = PiRGBArray(camera)
# allow the camera to warmup
time.sleep(0.1)

while True:
    # grab an image from the camera
    # camera.capture(rawCapture, format="bgr")
    # camera.capture(stream, f"data/frame-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg")

    image = np.empty((480 * 736 * 3,), dtype=np.uint8)
    camera.capture(image, 'bgr')
    image = image.reshape((480, 736, 3))


    # image = rawCapture.array
    # display the image on screen and wait for a keypress
    cv2.imwrite(f"data/frame-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg", image)
    time.sleep(5)