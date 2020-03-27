'''
@Description: 扫描答题卡，获取分数
@version: 
@Author: HLLI8
@Date: 2020-03-23 16:15:45
@LastEditors: HLLI8
@LastEditTime: 2020-03-24 19:34:58
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
ap.add_argument("-i", "--image", required=False, default="E:/PythonWorkSpace/DeepLearningWithOpenCV/Bubble_Sheet_Scanner/Week04Code/optical-mark-recognition/images/test_03.png", help="path to the input image")
args = vars(ap.parse_args())

#定义解答问题和正确答案的映射关系
ANSWER_KEY = {0:1, 1:4, 2:0, 3:3, 4:1}

#加载图片，灰度值化，轻微的模糊化，找边缘
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(blurred, 75, 200)

#找到边缘轮廓，初始化文档的边缘
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None

#确保至少一个边缘被找到
if len(cnts) > 0:
    #根据其大小用递减的方式排序
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    #loop over the sorted Contours
    for c in cnts:
        #估计边缘
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)

        #假设有四个点，就认为发现了paper
        if len(approx) == 4:
            docCnt = approx
            break

#将四点透视变换应用于原始图像和灰度图像，得到自上而下的鸟眼视图
paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))

#应用otsu的阈值法对扭曲的纸张进行二值化
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]

#在阈值图像中找到等高线，然后初始化对应于问题的等高线列表 
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []

for c in cnts:
    #计算轮廓的边界框，然后使用边界框导出纵横比
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    #为了把轮廓标记为一个问题，区域应该足够宽，足够高，并且有一个大约等于1的纵横比 
    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questionCnts.append(c)
    
#从上到下排序问题边缘，初始化整个问题的正确答案
questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
correct = 0

#每个问题有5个可能的答案
for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
    #从左到右为当前边缘排序，初始化圆圈答案的标签
    cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
    bubbled = None

    for (j, c) in enumerate(cnts):
        #construct a mask that reveals only the current "bubbled" for the question
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        #应用掩码到阈值图片，然后计算在圆区域中的非零像素数量
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)

        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)
        
    #初始化边缘颜色和正确答案的标签
    color = (0, 0, 255)
    k = ANSWER_KEY[q]

    #检查圆圈答案是否是正确
    if k == bubbled[1]:
        color = (0, 255, 0)
        correct += 1
    
    #在试卷图片中画出正确答案的下划线
    cv2.drawContours(paper, [cnts[k]], -1, color, 3)

score = (correct / 5.0) * 100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(paper, "{:.2f}%".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Original", image)
cv2.imshow("Exam", paper)
cv2.waitKey(0)