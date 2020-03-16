'''
@Description: 制作一个文档文字扫描器
@version: 
@Author: HLLI8
@Date: 2020-03-14 17:27:41
@LastEditors: HLLI8
@LastEditTime: 2020-03-16 15:00:05
'''
import sys
sys.path.append ("D:/ProgramFile/Anaconda/Lib/site-packages") 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #只显示warning和Error

from outsideLibrary.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

