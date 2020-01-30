'''
@Description: 学习OpenCV
@version: 
@Author: HLLI8
@Date: 2020-01-28 13:26:22
@LastEditors  : HLLI8
@LastEditTime : 2020-01-30 11:04:38
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

























