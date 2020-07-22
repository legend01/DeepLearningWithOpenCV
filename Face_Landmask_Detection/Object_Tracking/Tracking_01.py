'''
@Description: 
@version: 
@Author: HLLI8
@Date: 2020-07-22 10:00:48
@LastEditors: HLLI8
@LastEditTime: 2020-07-22 11:01:25
'''
import sys
sys.path.append ("D:/ProgramFile/Anaconda/Lib/site-packages") 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #只显示warning和Error

import dlib
import cv2

tracker = dlib.correlation_tracker() #导入correlation_tracker()类
cap = cv2.VideoCapture(0) #打开摄像头
start_flag = True #标记是否是第一帧
selection = None #实时跟踪鼠标的跟踪区域
track_window = None #要检测物体所在区域
drag_start = None #标记，是否开始拖动鼠标

#鼠标点击事件回调函数
def onMouseClicked(event, x, y, flags, param):
    global selection, track_window, drag_start #定义全局变量
    if event == cv2.EVENT_LBUTTONDOWN: #鼠标左键按下
        drag_start = (x, y)
        track_window = None
    if drag_start: #是否开始拖动鼠标，记录鼠标位置
        xMin = min(x, drag_start[0])
        yMin = min(y, drag_start[1])
        xMax = max(x, drag_start[0])
        yMax = max(y, drag_start[1])
        selection = (xMin, yMin, xMax, yMax)
    if event == cv2.EVENT_LBUTTONUP: #鼠标左键松开
        drag_start = None
        track_window = selection
        selection = None

if __name__ == "__main__":
    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("image", onMouseClicked)

    #opencv的bgr格式图片转换为rgb格式 bgr = cv2.split(frame)
    #frame2 = cv2.merge([r, g, b])
    while(True):
        ret, frame = cap.read() #从摄像头读取一帧
        
        if start_flag == True:
            #初始化，窗口停在当前帧，用鼠标拖拽一个框来指定区域,随后跟踪这个区域
            while True:
                imag_first = frame.copy() #不改变原来的帧，拷贝新的出来
                if track_window: #跟踪目标的窗口画不出来，实时标出来
                    print("[INFO] Track Object position:{}".format(track_window))
                    cv2.rectangle(imag_first, (track_window[0], track_window[1]), (track_window[2], track_window[3]), (0, 0, 255), 1)
                elif selection: #跟踪目标的窗口随鼠标拖动实时显示
                    print("[INFO] mouse potion:{}".format(selection))
                    cv2.rectangle(imag_first, (selection[0], selection[1]), (selection[2], selection[3]), (0, 0, 255), 1)
                cv2.imshow("image", imag_first)
                #按下回车退出循环
                if cv2.waitKey(5) == 13:
                    break
            start_flag = False #初始化完毕
            tracker.start_track(frame, dlib.rectangle(track_window[0], track_window[1], track_window[2], track_window[3])) #跟踪目标，目标就是选定目标窗口
        else:
            tracker.update(frame) #更新,实时跟踪
        
        box_predict = tracker.get_position() #得到目标位置
        print("[INFO] position:{}".format(box_predict))
        cv2.rectangle(frame, (int(box_predict.left()), int(box_predict.top())), (int(box_predict.right()), int(box_predict.bottom())), (0, 255, 0), 1) #用矩阵框标注出来
        cv2.imshow("image", frame)
        #如果按下ESC键，退出
        if cv2.waitKey(10) == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()