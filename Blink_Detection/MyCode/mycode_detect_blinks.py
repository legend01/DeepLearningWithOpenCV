'''
@Description: Detecting blinks with facial landmarks and opencv
@version: 
@Author: HLLI8
@Date: 2020-06-02 09:14:01
@LastEditors: HLLI8
@LastEditTime: 2020-06-02 09:18:13
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

