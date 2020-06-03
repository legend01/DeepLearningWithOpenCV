'''
@Description: Building the drowsiness detector with openCV
@version: 
@Author: HLLI8
@Date: 2020-06-03 14:37:19
@LastEditors: HLLI8
@LastEditTime: 2020-06-03 14:41:36
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

