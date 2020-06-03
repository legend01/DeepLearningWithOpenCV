'''
@Description: Building the drowsiness detector with openCV
@version: 
@Author: HLLI8
@Date: 2020-06-03 14:37:19
@LastEditors: HLLI8
@LastEditTime: 2020-06-03 15:00:24
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