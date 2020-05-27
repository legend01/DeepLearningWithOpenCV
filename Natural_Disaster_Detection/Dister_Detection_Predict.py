'''
@Description: 自然灾害监测
@version: 
@Author: HLLI8
@Date: 2020-05-21 17:50:06
@LastEditors: HLLI8
@LastEditTime: 2020-05-27 15:13:40
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

#从视频流中循环帧数据
while True:
    #读取文件中下一个帧数据
    (grabbed, frame) = vs.read()

    #如果没有摄取到视频帧数据,就是达到了视频流的末尾
    if not grabbed:
        break

    #如果视频帧维度为空,摄取
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    
    #关闭输出帧，从BGR格式到RGB顺序格式，裁剪帧为224*224
    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype("float32")

    #在数据帧中预测并且更新预测队列
    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    Q.append(preds)

    results = np.array(Q).mean(axis=0)
    i = np.argmax(results)
    label = config.CLASSES[i]

    