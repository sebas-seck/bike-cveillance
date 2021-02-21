#!/usr/bin/python3
# -*- encoding: utf-8 -*-
#
# https://picamera.readthedocs.io/en/release-1.13/recipes1.html
# https://cristianpb.github.io/blog/ssd-yolo

import json
import os
import click
import time
from datetime import datetime

import cv2
import numpy as np
from picamera import PiCamera

from detector import detect



# initialize the camera and grab a reference to the raw camera capture
# camera = PiCamera()
# # camera.resolution = (736, 480)
# camera.resolution = (736, 480)
# camera.hflip = True
# camera.vflip = True
# camera.led = False
# time.sleep(0.1)

@click.command()
@click.option("--save-all", is_flag=True, default=False)
@click.option("-r", "--resolution", default=(736,480))
@click.option("--crop", default=None, help="Tuple[minX,maxX,minY,maxY]")
@click.option("-i", "--intervall", type=click.INT, default=15, required=False)
def main(save_all, resolution, crop, intervall):
    camera = PiCamera()
    camera.resolution = (resolution[0], resolution[1])
    camera.shutter_speed = 5
    camera.hflip = True
    camera.vflip = True
    camera.led = False
    time.sleep(0.1)
    next_iter = time.time()  # solve with queues
    while True:
        if next_iter < time.time(): # solve with queues
            next_iter = time.time()+intervall # solve with queues

            # grab an image from the camera
            image_orig = np.empty((480 * 736 * 3,), dtype=np.uint8)
            camera.capture(image_orig, "rgb")
            image_orig = image_orig.reshape((resolution[1], resolution[0], 3))

            image_resized = image_orig[300:420,350:550].copy()

            # TODO turn detection model into cli arg
            pred_image, pred_df = detect(image_resized, "ssd_mobilenet")

            if save_all or len(pred_df) > 0:
                cv2.imwrite(f"data/frame-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_original.jpg", image_orig)
                cv2.imwrite(f"data/frame-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg", pred_image)

        else:  # solve with queues
            time.sleep(1)  # solve with queues

            
        

if __name__ == "__main__":
    main()