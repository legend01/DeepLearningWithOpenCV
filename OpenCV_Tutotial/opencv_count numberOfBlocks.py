'''
@Description: OpenCV计算俄罗斯方块
@version: 
@Author: HLLI8
@Date: 2020-03-14 11:59:52
@LastEditors: HLLI8
@LastEditTime: 2020-03-14 12:21:48
'''
""" 
@TODO: 1.利用OpenCV将图像转换为灰度
       2.执行边缘检测
       3.灰度图像阈值
       4.查找、计数和绘制等高线
       5.腐蚀和膨胀
       6.掩盖图像 
"""
import sys
sys.path.append ("D:/ProgramFile/Anaconda/Lib/site-packages") 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #只显示warning和Error
 
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Image", image)
cv2.waitKey(0)




























