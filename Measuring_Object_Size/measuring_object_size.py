'''
@Description: 通过计算机视觉测量物体的尺寸
@version: 
@Author: HLLI8
@Date: 2020-04-01 14:55:24
@LastEditors: HLLI8
@LastEditTime: 2020-04-01 14:58:04
'''
import sys
sys.path.append ("D:/ProgramFile/Anaconda/Lib/site-packages") 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #只显示warning和Error

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

