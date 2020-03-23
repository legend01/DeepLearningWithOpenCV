'''
@Description: 扫描答题卡，获取分数
@version: 
@Author: HLLI8
@Date: 2020-03-23 16:15:45
@LastEditors: HLLI8
@LastEditTime: 2020-03-23 16:24:21
'''
import sys
sys.path.append ("D:/ProgramFile/Anaconda/Lib/site-packages") 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #只显示warning和Error

from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

#定义解答问题和正确答案的映射关系
ANSWER_KEY = {0:1, 1:4, 2:0, 3:3, 4:1}
