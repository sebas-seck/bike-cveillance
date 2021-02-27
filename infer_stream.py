#!/usr/bin/python3
# -*- encoding: utf-8 -*-
#
# https://picamera.readthedocs.io/en/release-1.13/recipes1.html
# https://cristianpb.github.io/blog/ssd-yolo

import click
import time
from datetime import datetime

import cv2
import numpy as np
from picamera import PiCamera

from detector import detect


@click.command()
@click.option(
    "--save-all",
    is_flag=True,
    default=False,
    help="Save all images instead of images with a detected person.",
)
@click.option(
    "-r",
    "--resolution",
    type=(click.INT, click.INT),
    default=(1280, 1024),
    help="Resolution of the camera. Max resolution 2592x1944.",
)
@click.option(
    "--crop",
    type=(click.INT, click.INT, click.INT, click.INT),
    required=False,
    help="Crops image. Expects a tuple (minX,maxX,minY,maxY)",
)
@click.option(
    "-m",
    "--model",
    type=str,
    default="ssd_mobilenet",
    help="Model to be used, select from [ssd_mobilenet, yolo]",
)
@click.option(
    "-i",
    "--intervall",
    type=click.INT,
    default=15,
    required=False,
    help="Intervall between captures in seconds",
)
def main(save_all, resolution, crop, model, intervall):
    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (resolution[0], resolution[1])

    camera.hflip = True
    camera.vflip = True
    camera.led = False
    time.sleep(0.1)
    next_iter = time.time()  # solve with queues
    while True:
        if next_iter < time.time():  # solve with queues
            next_iter = time.time() + intervall  # solve with queues

            # grab an image from the camera
            image_orig = np.empty((resolution[0] * resolution[1] * 3,), dtype=np.uint8)
            camera.capture(image_orig, "rgb")
            image_orig = image_orig.reshape((resolution[1], resolution[0], 3))

            if crop:
                image_resized = image_orig[crop[0] : crop[1], crop[2] : crop[3]].copy()
            else:
                image_resized = image_orig

            # TODO turn detection model into cli arg
            pred_image, pred_df = detect(image_resized, model)
            print(pred_df)

            person = pred_df["class_name"].str.contains("person").any()
            if person:
                print(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - PERSON DETECTED!"
                )

            if save_all or person:
                cv2.imwrite(
                    f"data/frame-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_original.jpg",
                    image_orig,
                )
                cv2.imwrite(
                    f"data/frame-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg",
                    pred_image,
                )

        else:  # solve with queues
            time.sleep(1)  # solve with queues


if __name__ == "__main__":
    main()
