'''
@Description: 面部器官位置定位标记
@version: 
@Author: HLLI8
@Date: 2020-03-19 19:08:59
@LastEditors: HLLI8
@LastEditTime: 2020-03-19 19:17:10
'''
import sys
sys.path.append ("D:/ProgramFile/Anaconda/Lib/site-packages") 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #只显示warning和Error

from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_augment("-P", "--shape-predictor", required=True, help="path to facial landmarks predictor")
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

