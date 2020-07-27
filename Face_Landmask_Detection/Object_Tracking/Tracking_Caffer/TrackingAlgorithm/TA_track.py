'''
@Description: 
@version: 
@Author: HLLI8
@Date: 2020-07-27 10:19:08
@LastEditors: HLLI8
@LastEditTime: 2020-07-27 11:06:10
'''
import sys
sys.path.append ("D:/ProgramFile/Anaconda/Lib/site-packages") 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #只显示warning和Error

import cv2 
import sys

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if __name__ == '__main__':
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

    tracker_type = tracker_types[2]
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'NULL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if trakcer_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == 'CSRT':
            tracker = cv2.TrackerCSRT_create()

    video = cv2.VideoCapture("E:/PythonWorkSpace/DeepLearningWithOpenCV/Face_Landmask_Detection/Object_Tracking/Tracking_Caffer/multiobject-tracking-dlib/race.mp4")

    if not video.isOpened():
        print("[INFO] Could not open video")
        sys.exit()
    
    ok, frame = video.read()
    if not ok:
        print("[INFO] Could not read video file")
        sys.exit()
    
    bbox = (287, 23, 86, 320)
    bbox = cv2.selectROI(frame, False)

    ok = tracker.init(frame, bbox)

    while True:
        ok, frame = video.read()
        if not ok:
            break
        
        timer = cv2.getTickCount()
        
        ok, bbox = trakcer.update(frame)

        fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)

        if ok: 
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1]) + bbox[3])
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        
        cv2.imshow("Tracking", frame)

        k = cv2.waitKey(1) & oxff
        if k == 27: break

