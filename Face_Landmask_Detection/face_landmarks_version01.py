'''
@Description: 面部器官位置定位标记
@version: 
@Author: HLLI8
@Date: 2020-03-19 19:08:59
@LastEditors: HLLI8
@LastEditTime: 2020-03-19 21:07:50
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

#initialize dlib`s face detector (HOG-based) and then create the facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

#load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.Color_BGR2GRAY)

#detector faces in the grayscale image
''' 
@note: 在图片中处理面部边界
@Author: HLLI8
@Date: 2020-03-19 20:48:14
'''
rects = detector(gray, 1)

for (i, rect) in enumerate(rects):
    #面部区域检测面部标记，转换面部标记(x, y)坐标为numpy列表
    shape = predictor(gray, rect) 
    shape = face_utils.shape_to_np(shape)
    
    #转换dlib`s 矩形为openCV形式边界框[(x, y, w, h)]然后画出面部边界框
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #show the face number
    cv2.putText(image, "Face #{}".format(i +  1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #循环(x, y)坐标为面部区域做标记
    for(x, y) in shape:
        cv2.cicle(image, (x, y), 1, (0, 0, 255), -1)
    
#展示面部检测 + 面部区域标记图片输出
cv2.imshow("Output", image)
cv2.waitKey(0)

