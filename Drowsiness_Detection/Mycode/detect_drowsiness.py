'''
@Description: Building the drowsiness detector with openCV
@version: 
@Author: HLLI8
@Date: 2020-06-03 14:37:19
@LastEditors: HLLI8
@LastEditTime: 2020-06-03 16:39:23
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
    
    for rect in rects:
        shape = predictor(gray, rect) 
        shape = face_utils.shape_to_np(shape)
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1 
            
            #如果眼睛关闭在充足的次数后响起警报
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                #警报未打开，打开
                if not ALARM_ON:
                    ALARM_ON = True
                    
                    #检查警报文件是否支持，如果支持，开启一个线程播放报警
                    if args["alarm"] != "":
                        t = Thread(target=sound_alarm, args=(args["alarm"],))
                        t.deamon = True
                        t.start()
                
                #在图像中显示警报
                cv2.putText(frame, "DROWSINESS ALEART!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        else:
            COUNTER = 0
            ALARM_ON = False
        