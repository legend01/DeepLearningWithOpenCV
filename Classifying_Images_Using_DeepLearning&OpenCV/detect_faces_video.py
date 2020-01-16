'''
@Description: face detection in video and webcam with OpenCV and deep learning
@version: 
@Author: HLLI8
@Date: 2020-01-16 21:30:51
@LastEditors  : HLLI8
@LastEditTime : 2020-01-16 21:43:46
'''
#import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

#construct the arguments parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

#load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

#initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream....")
vs = VideoStream(src=0).start()
time.sleep(2.0)


























