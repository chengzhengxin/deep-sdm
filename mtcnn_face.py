# coding: utf-8
import mxnet as mx
import sys
sys.path.append('../mxnet_mtcnn_face_detection')
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time

def get_face_detector():
    detector = MtcnnDetector(model_folder='../mxnet_mtcnn_face_detection/model', ctx=mx.cpu(0), num_worker = 4 , accurate_landmark = False)
    return detector

def det_face(img):
    predout = detector.detect_face(img)
    # boxes  = predout[0]
    # points = predout[1]
    return predout