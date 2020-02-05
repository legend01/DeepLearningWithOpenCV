'''
@Description: 学习OpenCV
@version: 
@Author: HLLI8
@Date: 2020-01-28 13:26:22
@LastEditors  : HLLI8
@LastEditTime : 2020-02-05 21:52:30
'''
import sys
sys.path.append ("D:/ProgramFile/Anaconda/Lib/site-packages") 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #只显示warning和Error

#import the necessary packages
import imutils
import cv2

#load the input image and show its dimensions, keeping in mind that images are represented as a multi-dimensions NumPy array
#with shape no. rows(height) x no. colums (width) x no. channels (depth)
image = cv2.imread("E:/PythonWorkSpace/DeepLearningWithOpenCV/OpenCV_Tutotial/image/jp.png")
(h, w, d) = image.shape
print("width={}, height={}, depth={}".format(w, h, d))

#display the image to our screen --we will need to click the window open by OpenCV and press a key on our keyboard to continue execution
cv2.imshow("Image", image)
cv2.waitKey(0) #防止图像瞬间出现和消失


#在OpenCV中，图像的颜色标准顺序是BGR
#sccess the RGB pixel located at x=50, y=100, keeping in mind that OpenCV stores images in BGR order rather than RGB
(B, G, R) = image[100, 50]
print("R={}, G={}, B={}".format(R, G, B))

#extract a 100*100 pixel square ROI (Region of Interest) from the input image starting at x=320, y=60 at ending at x=420, y=160
roi = image[60:160, 320:420] #image[startY:endY, startX:endX]
cv2.imshow("ROI", roi)
cv2.waitKey(0)

#resize the iamge to 200*200px, ignoring aspect ratio
resized = cv2.resize(image, (200, 200))
cv2.imshow("Fixed Resizeding", resized)
cv2.waitKey(0)

#计算原始图像的纵横比，并使用它来调整图像的大小，使其不会出现扭曲
#fixed resizing and distort aspect ratio so let`s resize the width to be 300px but compute the new height based on the aspect ratio
r = 300.0/w #calculate the ratio of the new width to the old width
dim = (300, int(h * r)) #dimension is (width radio, height radio)
resized = cv2.resize(image, dim)
cv2.imshow("Aspect Ratio Resized", resized)
cv2.waitKey(0)

#manually computing the aspect ratio can be a pain so let`s use the
#imutils library instead
resized = imutils.resize(image, width=300) #利用OpenCV内部函数对图像尺寸进行放大或缩小
cv2.imshow("Imutils Resize", resized)
cv2.waitKey(0)

#let`s rotate an image 45 degrees clockwise using OpenCV by first
#computing the image center, then constructing the rotation matrix and then finally applying the affine warp
center = (w // 2, h // 2) #计算图像中心点 无浮点中心点
M = cv2.getRotationMatrix2D(center, -45, 1.0) #M表示仿射变化矩阵 1.0表示进行等比列的缩放
rotated = cv2.warpAffine(image, M, (w, h)) #(w, h)表示变换后的图片大小
cv2.imshow("OpenCV Rotation", rotated)
cv2.waitKey(0)

#rotation can also be easily accomplished via imutils with less code
rotated = imutils.rotate(image, -45)
cv2.imshow("Imutils Rotation", rotated)
cv2.waitKey(0)

#OpenCV doesn`t "care" if our rotated image is clipped after rotation
#so we can instead use another imutils conventionce function to help us out
rotated = imutils.rotate_bound(image, 45)
cv2.imshow("Imutils Bound Rotation", rotated)
cv2.waitKey(0)

#apply a Gaussian blur with a 11*11 kernel to the image to smooth it, useful when reducing high frequency noise
blurred = cv2.GaussianBlur(image, (11, 11), 0)
cv2.imshow("Blrred", blurred)
cv2.waitKey(0)
























