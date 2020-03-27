'''
@Description: 制作一个圆球追踪轨迹机器视觉程序
@version: 
@Author: HLLI8
@Date: 2020-03-25 18:07:00
@LastEditors: HLLI8
@LastEditTime: 2020-03-26 13:39:18
'''
import sys
sys.path.append ("D:/ProgramFile/Anaconda/Lib/site-packages") 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #只显示warning和Error

from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

#define the lower and upper boundaries of the "green" ball in the HSV color space, then initialize the list of tracked points
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
pts = deque(maxlen=args["buffer"])

#if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
    vs = VideoStream(src=0).start()
#otherwise, grab a refference to the video file
else:
    vs = cv2.VideoCapture(args["video"])
#allow the camera or video file to warm up
time.sleep(2.0)

while True:
    #grab the current frame
    frame = vs.read()

    #handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame

    #if we are viewing a video and we did not grab a frame, then we have reached the end of the video
    if frame is None:
        break
    
    #resize the frame, blur it, and convert it to the HSV color space
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    #construct a mask for the color "green", then perform a series of dilations and erosions to remove any small
    #blobs left in the mask
    mask = cv2.inRange(hav, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)