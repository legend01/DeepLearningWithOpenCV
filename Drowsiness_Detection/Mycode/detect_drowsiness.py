'''
@Description: Building the drowsiness detector with openCV
@version: 
@Author: HLLI8
@Date: 2020-06-03 14:37:19
@LastEditors: HLLI8
@LastEditTime: 2020-06-03 15:38:10
'''
import sys
sys.path.append ("D:/ProgramFile/Anaconda/Lib/site-packages") 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #只显示warning和Error

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2

def sound_alarm(path):
    # play an alarm sound
    playsound.playsound(path) 

def eye_aspect_ratio(eye):
    # 计算垂直坐标部分欧几里得距离
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # 计算水平坐标部分欧几里得距离
    C = dist.euclidean(eye[0], eye[3])

    # 计算纵横比
    ear = (A + B) / (2.0 * C)

    return ear

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape_predictor", required=True, help="path to facial landmarks prediction")
ap.add_argument("-a", "--alarm", type=str, default="", help="path alarm")
ap.add_argument("-w", "--webcam", type=int , default=0, help="index of webcam on system")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48

COUNTER = 0
ALARM_ON = False

print("[INFO] loading facial landmarks predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 0)