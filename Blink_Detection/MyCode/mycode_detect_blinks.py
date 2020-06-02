'''
@Description: Detecting blinks with facial landmarks and opencv
@version: 
@Author: HLLI8
@Date: 2020-06-02 09:14:01
@LastEditors: HLLI8
@LastEditTime: 2020-06-02 14:09:09
'''
import sys
sys.path.append ("D:/ProgramFile/Anaconda/Lib/site-packages") 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #只显示warning和Error

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

def eye_aspect_ratio(eye):
    #计算垂直坐标之间的欧几里得距离
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    #计算水平眼镜区域之间的欧几里得距离
    C = dist.euclidean(eye[0], eye[3])
    
    #计算眼睛的纵横比
    ear = (A + B)/(2.0 * C)

    #返回眼睛的纵横比
    return ear

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmarks predictor")
ap.add_argument("-v", "--video", type=str, default="", help="path to input video file")
args = vars(ap.parse_args())

# 眼睛长宽比低于某一阈值，然后上升到阈值以上，注册一个“眨眼”
EYE_AR_THRESH = 0.25
# 指示三个连续帧的眼睛纵横比小于eye_ar_thresh必须发生才能注册眨眼
EYE_AR_CONSEC_FRAMES = 6

COUNTER = 0
TOTAL = 0

print("[INFO] loading facial landmarks predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

#获取左右边面部眼睛的位置
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()
fileStream = True
time.sleep(2.0)

while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        # 计算双眼的纵横比
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
        
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            
            COUNTER = 0

        # 在图像中写出眨眼的次数和纵横比
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 显示图像
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # q按键退出循环
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
