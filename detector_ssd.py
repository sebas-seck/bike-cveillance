import os
import json
import cv2
import numpy as np
import pandas as pd
from utils import draw_boxed_text


class Detector():
    """Class ssd"""

    def __init__(self, config):
        self.model = cv2.dnn.readNetFromTensorflow(
            config["MODEL_WEIGHTS"],
            config["MODEL_CONFIG"],
                )
        self.colors = np.random.uniform(0, 255, size=(100, 3))
        self.config = config

    def prediction(self, image):
        self.model.setInput(
                cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=self.config["SWAPRB"]))
        output = self.model.forward()
        result = output[0, 0, :, :]
        return result

    def filter_prediction(self, output, image, conf_th=0.5, conf_class=[]):
        height, width = image.shape[:-1]
        df = pd.DataFrame(
                output,
                columns=[
                    '_', 'class_id', 'confidence', 'x1', 'y1', 'x2', 'y2'])
        df = df.assign(
                x1=lambda x: (x['x1'] * width).astype(int).clip(0),
                y1=lambda x: (x['y1'] * height).astype(int).clip(0),
                x2=lambda x: (x['x2'] * width).astype(int),
                y2=lambda x: (x['y2'] * height).astype(int),
                class_name=lambda x: (
                    x['class_id'].astype(int).astype(str).replace(self.config["CLASS_NAMES"])
                    ),
                )
        df['label'] = (df['class_name'] + ': ' +
                       df['confidence'].astype(str).str.slice(stop=4))
        df = df[df['confidence'] > conf_th]
        if len(conf_class) > 0:
            df = df[df['class_id'].isin(conf_class)]
        return df

    def draw_boxes(self, image, df):
        for idx, box in df.iterrows():
            x_min, y_min, x_max, y_max = box['x1'], box['y1'], box['x2'], box['y2']
            color = self.colors[int(box['class_id'])]
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            txt_loc = (max(x_min+2, 0), max(y_min+2, 0))
            txt = box['label']
            image = draw_boxed_text(image, txt, txt_loc, color)
        return image