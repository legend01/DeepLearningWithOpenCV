'''
@Description: Detecting blinks with facial landmarks and opencv
@version: 
@Author: HLLI8
@Date: 2020-06-02 09:14:01
@LastEditors: HLLI8
@LastEditTime: 2020-06-02 09:26:43
'''
import sys
sys.path.append ("D:/ProgramFile/Anaconda/Lib/site-packages") 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #只显示warning和Error

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

def eye_aspect_ratio(eye):
    #计算垂直坐标之间的欧几里得距离
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    #计算水平眼镜区域之间的欧几里得距离
    C = dist.euclidean(eye[0], eye[3])
    
    #计算眼睛的纵横比
    ear = (A + B)/(2.0 * C)

    #返回眼睛的纵横比
    return ear

