'''
@Description: 自然灾害监测
@version: 
@Author: HLLI8
@Date: 2020-05-21 17:50:06
@LastEditors: HLLI8
@LastEditTime: 2020-05-22 18:26:37
'''
import sys
sys.path.append ("D:/ProgramFile/Anaconda/Lib/site-packages") 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #只显示warning和Error

from tensorflow.keras.models import load_model
from Outside_Library import config
from collections import deque
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to our input video")
ap.add_argument("-o", "--output", required=True, help="path to our output video")
ap.add_argument("-s", "--size", type=int, default=128, help="size of queue for averaging")
ap.add_argument("-d", "--display", type=int, default=-1, help="whether or not output frame should be displayed to screen")
args = vars(ap.parse_args())

#加载预训练模型
print("[INFO] loading model and label binarizer....")
model = load_model(config.MODEL_PATH)

#初始化预定义队列
Q = deque(maxlen=args['size'])

#初始化视频流,指出输出视屏文件夹，视屏规格
print("[INFO] processing video....")
vs = cv2.VideoCapture(args["input"])
wirte = None
(W, H) = (None, None)