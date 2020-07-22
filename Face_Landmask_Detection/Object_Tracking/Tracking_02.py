'''
@Description: 
@version: 
@Author: HLLI8
@Date: 2020-07-22 13:45:11
@LastEditors: HLLI8
@LastEditTime: 2020-07-22 22:53:07
'''
import sys
sys.path.append ("D:/ProgramFile/Anaconda/Lib/site-packages") 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #只显示warning和Error

import dlib
import cv2

class CorrelationTracker(Object):
    def __init__(self, windowName='default window', cameraNum=0):
        self.STATUS_RUN_WITHOUT_TRACKER = 0 #不跟踪，实时显示
        self.STATUS_RUN_WITH_TRACKER = 1 #跟踪目标，实时显示
        self.STATUS_PAUSE = 2 #暂停，当前帧
        self.STATUS_BREAK = 3 #退出
        self.status = self.STATUS_SUN_WITHOUT_TRACKER #指示状态的变量

        self.track_window = None #实时跟踪鼠标的跟踪区域
        self.drag_start = None #检测的物体所在区域
        self.start_flag = True #标记，是否开始拖动鼠标

        # 创建显示窗口
        cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(windowName, self.onMouseClicked)
        self.windowName = windowName

        self.cap = cv2.VideoCapture(cameraNum)
        
        #correlation_tracker()类，跟踪器
        self.tracker = dlib.correlation_tracker()

        # 当前帧
        self.frame = None

    def keyEventHandler(self):
        keyValue = cv2.waitKey(5) #每隔5ms读取一次按键键值
        if keyValue == 27: #ESC
            self.status = self.STATUS_BREAK
        if keyValue == 32: #空格
            if self.status != self.STATUS_PAUSE: #按空格，暂停播放，可选定跟踪区域
                self.status = self.STATUS_PAUSE
            else: # 再次空格，重新播放，但不进行目标识别
                if self.track_window:
                    self.status = self.STATUS_RUN_WITH_TRACKER
                    self.start_flag = True
                else:
                    self.status = self.STATUS_RUN_WITHOUT_TRACKER
        if keyValue == 13: #回车
            if self.status == self.STATUS_PAUSE: #按下空格后
                if self.track_window: #如果选定区域，再按回车，表示确定选定区域为跟踪目标
                    self.status = self.STATUS_RUN_WITH_TRACKER
                    self.start_flag = True
                
    def processHandler(self):
        # 不跟踪目标，实时显示
        if self.status == self.STATUS_RUN_WITHOUT_TRACKER:
            ret, self.frame = self.cap.read()
            cv2.imshow(self.windowName, self.frame)
        # 暂停，使用鼠标拖动红框，选择目标区域
        elif self.status == self.STATUS_PAUSE:
            img_first = self.frame.copy() #不改变原来帧，拷贝
            if self.track_window:
                cv2.rectangle(img_first, (self.track_window[0], self.track_window[1]), (self.track_window[2], self.track_window[3])) #开始跟踪目标
            elif self.selection: 
                cv2.rectangle(img_first, (self.selection[0], self.selection[1]), (self.selection[2], self.selection[3]), (0, 0, 255), 1)
            cv2.imshow(self.windowName, img_first)
        # 退出
        elif self.status == self.STATUS_BREAK:
            self.cap.release() 
            cv2.destroyAllWindows()
            sys.exit()
        # 跟踪目标，实时显示
        elif self.status == self.STATUS_RUN_WITH_TRACKER:
            ret, self.frame = self.cap.read()
            if self.start_flag:
                self.tracker.start_track(self.frame, dlib.rectangle(self.track_window[0], self.track_window[1], self.track_window[2], self.track_window[3]))  
                self.start_flag = False  
            else:
                self.tracker.update(self.frame)
                # 得到目标位置，并显示
                box_predict = self.tracker.get_position()
                cv2.rectangle(self.frame, (int(box_predict.left()), int(box_predict.top()), int(box_predict.right()), int(box_predict.bottom()), (0, 255, 255), 1)
                cv2.imshow(self.windowName, self.frame)

    def onMouseClicked(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN: #鼠标左键
            self.drag_start = (x, y)
            self.track_window = None
        if self.drag_start: #是否开始拖动鼠标，记录鼠标位置
            xMin = min(x, self.drag_start[0])
            yMin = min(y, self.drag_start[1])
            xMax = max(x, self.drag_start[0])
            yMax = max(y, self.drag_start[1])
            self.selection = (xMin, yMin, xMax, yMax)
        if event == cv2.EVENT_LBUTTONUP: #鼠标左键松开
            self.drag_start = None
            self.track_window = self.selection
            self.selection = None
    
    def run(self):
        while(1):
            self.keyEventHandler()
            self.processHandler()

if __name__ == "__main__":
    testTracker = CorrelationTracker(windowName="Object Tracking", cameraNum=0)
    testTracker.run()
            