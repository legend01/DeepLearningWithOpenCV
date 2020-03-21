'''
@Description: 在视频流中检测面部区域
@version: landmarks_vedioStream_version01
@Author: HLLI8
@Date: 2020-03-20 15:41:47
@LastEditors: HLLI8
@LastEditTime: 2020-03-21 11:46:57
'''
import sys
sys.path.append ("D:/ProgramFile/Anaconda/Lib/site-packages") 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #只显示warning和Error

from imutils import face_utils
from imutils.video import VideoStream
import argparse
import imutils 
import dlib
import time
import cv2

def CatchVideoStream():
    # cap = cv2.VideoCapture(0)
    # while(1):
    #     ret, frame = cap.read()
    #     cv2.imshow("capture", frame)
    #     if cv2.waitKey(1) & 0xFF == ord("q"): #q键完成抓取图像帧
    #         return frame #返回图像
    print("[INFO] starting video stream........")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    
    return vs #返回视屏流
    # cap.release()
    # cv2.destroyAllWindows()

def main():
    #设置所需参数
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=False, help="path to facial landmark prediction")
    predictor_path = "E:/PythonWorkSpace/DeepLearningWithOpenCV/Face_Landmask_Detection/Gradient_Descent_DS/shape_predictor_68_face_landmarks.dat"
    ap.add_argument("-i", "--image", required=False, default='0', help="path to input image")
    #image_path = "E:/PythonWorkSpace/DeepLearningWithOpenCV/Face_Landmask_Detection/image/two_people.jpg"
    args = vars(ap.parse_args())

    #初始化dlib人脸检测，创建面部标志预测器
    detector = dlib.get_frontal_face_detector()
    #predictor = dlib.shape_predictor(args["shape_predictor"])
    predictor = dlib.shape_predictor(predictor_path)

    if args['image'] != '0':
        #if image_path != ' ':
            #image = cv2.imread(args['image']) #输入图片实参读入图片
            image = cv2.imread(image_path)
    else: 
        # image = takephoto() #若未输入则进行照相操作
        vs = CatchVideoStream()

    while True:
        frame = vs.read()
        image = imutils.resize(frame, width=800) #调整图片宽度500
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

        #检测灰度图像中的面部
        rects = detector(gray, 1)

        for (i, rect) in enumerate(rects):
            #确定面部区域面部标志，将面部标志(x, y)转换为Numpy阵列
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            #将dlib矩形转换成OpenCV样式边界框[(x, y, w, h)],绘制边界框
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            #人脸序号标记
            cv2.putText(image, "Face #{}".format(i+1), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            print("[INFO] Face tagging......")
            #循环找到面部关键点(x, y)坐标并在图像中绘制
            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF

        if (key == ord('b')):
            break
        
        # #用脸部检测+面部标志显示输出图像
        # cv2.imshow("Output", image)
        # cv2.waitKey(0)
    cv2.destroyAllWindows()
    vs.stop()
    
if __name__ == "__main__":
    main()
