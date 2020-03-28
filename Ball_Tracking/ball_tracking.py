'''
@Description: 制作一个圆球追踪轨迹机器视觉程序
@version: 
@Author: HLLI8
@Date: 2020-03-25 18:07:00
@LastEditors: HLLI8
@LastEditTime: 2020-03-28 17:58:17
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
#greenLower = (29, 86, 6)
#greenUpper = (64, 255, 255)
greenLower = (170, 100, 100)
greenUpper = (179, 255, 255)
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
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    #find contours in the mask and initialize the current (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    #only proceed if at least one contour was found
    if len(cnts) > 0:
        #find the largest contour int the mask, then use it to compute the minimum enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c) #找到最小外接圆
        M = cv2.moments(c)
        center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])) #计算图像的质心

        #only proceed if the radius meets a minimum size
        if radius > 10: #如果找到轮廓的圆形大于10
            #draw the cicle and centroid on the frame, then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2) #围着检测的物体画圆
            cv2.circle(frame, center, 5, (0, 0, 255), -1) #画出质心
        
    #update the points queue 把质心添加到pts中，并且是添加到列表左侧
    pts.appendleft(center)  #遍历追踪点，分段画出轨迹 
    
    #loop over the set of tracked points 遍历追踪点，分段画出轨迹
    for i in range(1, len(pts)):
        #if either of the tracked pints are None, ignore them
        if pts[i - 1] is None or pts[i] is None:
            continue
        
        #otherwise, compute the thickness of the line and draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5) #计算所画小线段的粗细
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness) #画出小线段 

    #show the frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    #if the "q" key is pressed, stop the loop
    if key == ord("q"):
        break

#if we are not using a video file, stop the camera video streams
if not args.get("video", False):
    vs.stop()
#otherwise, release the camera
else:
    vs.release()

#close all windows
cv2.destroyAllWindows()