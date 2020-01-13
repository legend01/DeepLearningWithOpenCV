'''
@Description: We`ll be creating a Python script that can be used to classify input images
using OpenCV and GoogleLeNet(pre-trained on ImageNet) using the Caffe framework.
@version: 
@Author: HLLI8
@Date: 2020-01-13 21:41:01
@LastEditors  : HLLI8
@LastEditTime : 2020-01-13 22:02:07
'''
#import the necessary packages
import numpy as np
import argparse
import time
import cv2

#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-l", "--label", required=True, help="path to ImageNet labels(i.e., syn-sets)")
args = vars(ap.parse_args())






















