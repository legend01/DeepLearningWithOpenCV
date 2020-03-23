'''
@Description: 扫描答题卡，获取分数
@version: 
@Author: HLLI8
@Date: 2020-03-23 16:15:45
@LastEditors: HLLI8
@LastEditTime: 2020-03-23 17:41:07
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

#加载图片，灰度值化，轻微的模糊化，找边缘
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(blurred, 75, 200)

#找到边缘轮廓，初始化文档的边缘
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_AIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None

#确保至少一个边缘被找到
if len(cnts) > 0:
    #根据其大小用递减的方式排序
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    #loop over the sorted Contours
    for c in cnts:
        #估计边缘
        preri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)

        #假设有四个点，就认为发现了paper
        if len(approx) == 4:
            docCnt = approx
            break

#将四点透视变换应用于原始图像和灰度图像，得到自上而下的鸟眼视图
paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))
