'''
@Description: OpenCV计算俄罗斯方块
@version: 
@Author: HLLI8
@Date: 2020-03-14 11:59:52
@LastEditors: HLLI8
@LastEditTime: 2020-03-14 14:11:41
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
ap.add_argument("-i", "--image", required=False, help="path to input image")
#args = vars(ap.parse_args())
imgpath = "E:/PythonWorkSpace/DeepLearningWithOpenCV/OpenCV_Tutotial/image/tetris_blocks.png"

#image = cv2.imread(args["image"])
image = cv2.imread(imgpath)
cv2.imshow("Image", image)
cv2.waitKey(0)

#convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("GRAY", gray)
cv2.waitKey(0)

#applying edge detection we can find the outlines of objects in images
'''
@name: cv2.Canny()
@brief: 检测图像的边缘
@param: img --> 灰度图相关
        minVal --> 最小的阈值， 30
        maxVal --> 最大的阈值， 150
        aperture_size --> 贝索尔内核大小 默认情况 3
@return: 
@note: 
@Author: HLLI8
@Date: 2020-03-14 12:41:41
'''
edged = cv2.Canny(gray, 30, 150)
cv2.imshow("Edged", edged)
cv2.waitKey(0)

#threshold the image by setting all pixel values less than 225 to 255 and all pixel values >= 225 to 255, thereby segmenting the image.
thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)























