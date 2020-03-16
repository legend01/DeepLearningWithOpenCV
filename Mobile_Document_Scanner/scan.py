'''
@Description: 制作一个文档文字扫描器
@version: 
@Author: HLLI8
@Date: 2020-03-14 17:27:41
@LastEditors: HLLI8
@LastEditTime: 2020-03-16 16:04:24
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

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False, help="Path to the image to be scanned")
args = vars(ap.parse_args())
image_path = "E:/PythonWorkSpace/DeepLearningWithOpenCV/Mobile_Document_Scanner/image/receipt.jpg"
#load the image and compute the ratio of the old height to the new height, clone it, and resize it
#image = cv2.imread(args["image"])
image = cv2.imread(image_path)
ratio = image.shape[0] / 500
orig = image.copy()
image = imutils.resize(image, height=500)

#convert the image to grayscale, blur it, and find edges in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

#show the original image and the edge detected image
print("STEP 1: Edge Detection")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()
