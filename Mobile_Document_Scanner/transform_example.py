'''
@Description: 实验外部依赖文件transform.py中的图像映射转换算法
@version: 
@Author: HLLI8
@Date: 2020-03-16 14:58:08
@LastEditors: HLLI8
@LastEditTime: 2020-03-16 15:08:31
'''
import sys
sys.path.append ("D:/ProgramFile/Anaconda/Lib/site-packages") 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #只显示warning和Error
 
from outsideLibrary.transform import four_point_transform
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image file")
ap.add_argument("-c", "--coords", help="comma seperated list of source points")
args = vars(ap.parse_args())


