'''
@Description: We`ll be creating a Python script that can be used to classify input images
using OpenCV and GoogleLeNet(pre-trained on ImageNet) using the Caffe framework.
@version: 
@Author: HLLI8
@Date: 2020-01-13 21:41:01
@LastEditors  : HLLI8
@LastEditTime : 2020-01-16 20:40:23
'''
import sys
sys.path.append ("D:/ProgramFile/Anaconda/Lib/site-packages") 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #只显示warning和Error

#import the necessary packages
import numpy as np
import argparse
import cv2

#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

#load our serialized model from disk
print("[INFO] loading model...")
prototxt = "E:/PythonWorkSpace/DeepLearningWithOpenCV/Classifying_Images_Using_DeepLearning&OpenCV/model/deploy.prototxt.txt"
model = "E:/PythonWorkSpace/DeepLearningWithOpenCV/Classifying_Images_Using_DeepLearning&OpenCV/model/res10_300x300_ssd_iter_140000.caffemodel"
image = "E:/PythonWorkSpace/DeepLearningWithOpenCV/Classifying_Images_Using_DeepLearning&OpenCV/image/iron_chic.jpg"
confidence_threshold = 0.5
#net = cv2.dnn.readNetFromCaffe(args["prototxt"], arg["model"])
net = cv2.dnn.readNetFromCaffe(prototxt, model)

#load the input image and construct and input blob for the image by resizing to a fixed 300*300
#pixels and then normalizing it
#image = cv2.imread(args["image"])
image = cv2.imread(image)
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

#pass the blob through the network and obtain the detections and
#predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

#loop over the detections
for i in range(0, detections.shape[2]):
    #extract the confidence associated with the prediction
    confidence = detections[0, 0, i, 2]

    #filter out weak detections by ensuring the 'confidence' is greater than
    #the minimum confidence
    #if confidence > args["confidence"]:
    if confidence > confidence_threshold:
        #compute the (x, y)-coordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        #draw the bounding box of the face along with the associated probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
#show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)













