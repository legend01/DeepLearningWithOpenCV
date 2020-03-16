'''
@Description: 实验外部依赖文件transform.py中的图像映射转换算法
@version: 
@Author: HLLI8
@Date: 2020-03-16 14:58:08
@LastEditors: HLLI8
@LastEditTime: 2020-03-16 15:27:39
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
img_path = "E:/PythonWorkSpace/DeepLearningWithOpenCV/Mobile_Document_Scanner/Week03Code/document-scanner/images/page.jpg"
four_cordinate = "[(101, 185), (393, 151), (479, 323), (187, 441)]"
#加载图片并且抓取源地址
#NOTE：using the 'eval' function is bad form, but for this example let`s just roll with it
#in future posts I`ll show you how to automatically determine the coordinates without pre-supplying them
#image = cv2.imread(args["image"])
image = cv2.imread(img_path)
#pts = np.array(eval(args["coords"]), dtype="float32")
pts = np.array(eval(four_cordinate), dtype="float32")
#apply the four point transform to obtrain a "birds eye view" of the image
werped = four_point_transform(image, pts)

#show the orginal and warped images
cv2.imshow("Original Image", image)
cv2.imshow("Warped Image", werped)
cv2.waitKey(0)

