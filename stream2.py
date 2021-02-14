#!/usr/bin/python3
# -*- encoding: utf-8 -*-
#
# https://picamera.readthedocs.io/en/release-1.13/recipes1.html
# https://cristianpb.github.io/blog/ssd-yolo

import json
import os
import time
from datetime import datetime

import cv2
import numpy as np
# import pandas as pd
from picamera import PiCamera
from detector_yolo import Detector

from config import full_config

config = full_config["yolo"]

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (736, 480)
camera.hflip = True
camera.vflip = True
camera.led = False
# allow the camera to warmup
time.sleep(0.1)

while True:
    # grab an image from the camera
    image = np.empty((480 * 736 * 3,), dtype=np.uint8)
    camera.capture(image, "bgr")
    image = image.reshape((480, 736, 3))

    detector = Detector(config)
    output = detector.prediction(image)
    df = detector.filter_prediction(output, image)
    print(df)
    image = detector.draw_boxes(image, df)

    cv2.imwrite(f"data/frame-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg", image)
    # time.sleep(5)
