'''
@Description: face detection in video and webcam with OpenCV and deep learning
@version: 
@Author: HLLI8
@Date: 2020-01-16 21:30:51
@LastEditors  : HLLI8
@LastEditTime : 2020-01-17 22:14:55
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

#loop over the frames from the video stream
while True:
    #grab the frame from the threaded video stream and resize it
    #to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    #grap the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    #pass the blob through the network and obtain the detection and predictions
    net.setInput(blob)
    detection = net.forward()
    






















