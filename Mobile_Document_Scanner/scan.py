'''
@Description: 制作一个文档文字扫描器
@version: 
@Author: HLLI8
@Date: 2020-03-14 17:27:41
@LastEditors: HLLI8
@LastEditTime: 2020-03-16 17:32:37
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

#find the contours in the edged image, keeping only the largest ones, and initialize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5] #检测最大的边缘

#loop over the contours
for c in cnts:
    #approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    #如果大概的边缘有四个点，就能假设已经发现图像范围
    if len(approx) == 4:
        screenCnt = approx
        break

#显示文本的边缘线
print("STEP 2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#apply the four point transform to obtain a top-down view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2)*ratio)

#转换映射的图片为灰色，使其成为灰白效果
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset=10, method="gaussian")
warped = (warped > T).astype("uint8") * 255

#显示原始和扫描的图片
print("STEP 3: Apply perspective transform")
cv2.imshow("Original Image", imutils.resize(orig, height = 650))
cv2.imshow("Scanned Image", imutils.resize(warped, height = 650))
cv2.waitKey(0)
