"""
Face Detection Utility
Uses OpenCV's DNN-based face detector (ResNet-10 SSD).

Download model files:
    prototxt : https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
    caffemodel: https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
"""

import cv2
import numpy as np
import os
import urllib.request


PROTOTXT_URL  = ("https://raw.githubusercontent.com/opencv/opencv/master/"
                 "samples/dnn/face_detector/deploy.prototxt")
CAFFEMODEL_URL = ("https://github.com/opencv/opencv_3rdparty/raw/"
                  "dnn_samples_face_detector_20170830/"
                  "res10_300x300_ssd_iter_140000.caffemodel")


def download_face_detector(save_dir="face_detector"):
    """Auto-download face detector weights if missing."""
    os.makedirs(save_dir, exist_ok=True)
    proto_path = os.path.join(save_dir, "deploy.prototxt")
    model_path = os.path.join(save_dir, "res10_300x300_ssd_iter_140000.caffemodel")

    if not os.path.exists(proto_path):
        print("[FaceDetector] Downloading deploy.prototxt ...")
        urllib.request.urlretrieve(PROTOTXT_URL, proto_path)

    if not os.path.exists(model_path):
        print("[FaceDetector] Downloading caffemodel ...")
        urllib.request.urlretrieve(CAFFEMODEL_URL, model_path)

    return proto_path, model_path


def load_face_detector(proto_path, model_path):
    """Load OpenCV DNN face detector."""
    net = cv2.dnn.readNet(proto_path, model_path)
    return net


def detect_faces(net, frame, conf_threshold=0.5):
    """
    Detect faces in a BGR frame.

    Returns:
        List of (startX, startY, endX, endY) bounding boxes
    """
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                  (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")
            # Clip to frame boundaries
            startX, startY = max(0, startX), max(0, startY)
            endX,   endY   = min(w - 1, endX), min(h - 1, endY)
            boxes.append((startX, startY, endX, endY))

    return boxes
