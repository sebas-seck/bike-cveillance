#!/usr/bin/python3
# -*- encoding: utf-8 -*-

import cv2
from detector import detect

# TODO turn hardcode into option
image = cv2.imread("data/raw/street_cars.jpg")

pred_image, pred_df = detect(image, "ssd_mobilenet")

cv2.imwrite(f"data/raw/test.jpg", image)