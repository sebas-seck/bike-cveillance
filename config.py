import os
import json


# YOLO

full_config = {}

full_config["yolo"] = {
    "DETECTION_MODEL": "yolo",
    "THRESHOLD": 0.3,
    "SCALE": 0.00392,  # 1/255
    "NMS_THRESHOLD": 0.4,  # Non Maximum Supression threshold
    "SWAPRB": True,
    "CLASS_NAMES": None,
    "MODEL_CONFIG": "models/yolo/yolov3-tiny.cfg",
    "MODEL_WEIGHTS": "models/yolo/yolov3-tiny.weights",
    "RESOLUTION": (416, 416)
    # "MODEL_CONFIG": "models/yolo/yolov3.cfg",
    # "MODEL_WEIGHTS": "models/yolo/yolov3.weights",
    # "RESOLUTION": (608, 608)
}

with open(os.path.join("./models", "yolo", "labels.json")) as json_data:
    full_config["yolo"]["CLASS_NAMES"] = json.load(json_data)

# SSD

full_config["ssd_mobilenet"] = {
    "DETECTION_MODEL": "ssd_mobilenet",
    "SWAPRB": True,
    "MODEL_CONFIG": "models/ssd_mobilenet/ssd_mobilenet_v2_coco_2018_03_29.pbtxt",
    "MODEL_WEIGHTS": "models/ssd_mobilenet/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb",
}

with open(os.path.join('models', "ssd_mobilenet", 'labels.json')) as json_data:
    full_config["ssd_mobilenet"]["CLASS_NAMES"] = json.load(json_data)