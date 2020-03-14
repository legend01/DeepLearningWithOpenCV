'''
@Description: OpenCV计算俄罗斯方块
@version: 
@Author: HLLI8
@Date: 2020-03-14 11:59:52
@LastEditors: HLLI8
@LastEditTime: 2020-03-14 16:02:54
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

#find contours of the foreground objects in the threshold image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = image.copy()

#loop over the contours
for c in cnts:
    #draw each contour on the output image with a 3px thick purple outline, then display the output contours one at a time
    cv2.drawContours(output, [c], -1, (240, 0, 159), 3)
    cv2.imshow("Conrours", output)
    cv2.waitKey(0)

#draw the total number of contours found in purple
text = "I found {} objects!".format(len(cnts))
cv2.putText(output, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 0, 159), 2)
cv2.imshow("Contours", output)
cv2.waitKey(0)

#we apply erosions to reduce the size of foreground objects
mask = thresh.copy()
mask = cv2.erode(mask, None, iterations=10)
cv2.imshow("Eroded", mask)
cv2.waitKey(0)

#similarly, dilations can increase the size of the ground object
mask = thresh.copy()
mask = cv2.dilate(mask, None, iterations=5)
cv2.imshow("Dilated", mask)
cv2.waitKey(0)

#a typical operation we may want to apply is to take our mask and apply a bitwise AND to our input aimge, keeping only the masked regions
mask = thresh.copy()
output = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Output", output)
cv2.waitKey(0)











