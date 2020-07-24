import os
import glob
import cv2
import dlib

# Path to the video frames
video_folder = os.path.join("..", "examples", "video_frames2")

# Create the correlation tracker - the object needs to be initialized before it can be used
tracker1 = dlib.correlation_tracker()
tracker2 = dlib.correlation_tracker()
tracker3 = dlib.correlation_tracker()

selection = None
track_window = None
drag_start = None

def onmouse(event, x, y, flags, param):
    global selection,track_window,drag_start
    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = (x, y)
        track_window = None
    if drag_start:
        xmin = min(x, drag_start[0])
        ymin = min(y, drag_start[1])
        xmax = max(x, drag_start[0])
        ymax = max(y, drag_start[1])
        selection = (xmin, ymin, xmax, ymax)
    if event == cv2.EVENT_LBUTTONUP:
        drag_start = None
        track_window = selection
        selection = None

def main():
    track_window1 = ()
    track_window2 = ()
    track_window3 = ()
    cv2.namedWindow('image',1)
    cv2.setMouseCallback('image',onmouse)
    # We will track the frames as we load them off of disk
    for k, f in enumerate(sorted(glob.glob(os.path.join(video_folder, "*.jpg")))):
        print("Processing Frame {}".format(k))
        img_raw = cv2.imread(f)
        image = img_raw.copy()

        # We need to initialize the tracker on the first frame
        if k == 0:
            # Start a track on the object you want. box the object using the mouse and press 'Enter' to start tracking  
            while True:
                img_first = image.copy()
                if track_window:
                    cv2.rectangle(img_first,(track_window[0],track_window[1]),(track_window[2],track_window[3]),(0,0,255),1)
                elif selection:
                    cv2.rectangle(img_first,(selection[0],selection[1]),(selection[2],selection[3]),(0,0,255),1)

                if track_window1:
                    cv2.rectangle(img_first,(track_window1[0],track_window1[1]),(track_window1[2],track_window1[3]),(0,255,255),1)
                if track_window2:
                    cv2.rectangle(img_first,(track_window2[0],track_window2[1]),(track_window2[2],track_window2[3]),(0,255,100),1)
                if track_window3:
                    cv2.rectangle(img_first,(track_window3[0],track_window3[1]),(track_window3[2],track_window3[3]),(200,0,200),1)
                cv2.imshow('image',img_first)
                if cv2.waitKey(10) == 10:
                    if not track_window1:
                        track_window1 = track_window
                    elif not track_window2:
                        track_window2 = track_window
                    elif not track_window3:
                        track_window3 = track_window
                    else:
                        break
            tracker1.start_track(image, dlib.rectangle(track_window1[0], track_window1[1], track_window1[2], track_window1[3]))
            tracker2.start_track(image, dlib.rectangle(track_window2[0], track_window2[1], track_window2[2], track_window2[3]))
            tracker3.start_track(image, dlib.rectangle(track_window3[0], track_window3[1], track_window3[2], track_window3[3]))
        else:
            # Else we just attempt to track from the previous frame
            tracker1.update(image)
            tracker2.update(image)
            tracker3.update(image)

        # Get previous box and draw on showing image
        box1_predict = tracker1.get_position()
        box2_predict = tracker2.get_position()
        box3_predict = tracker3.get_position()
        cv2.rectangle(image,(int(box1_predict.left()),int(box1_predict.top())),(int(box1_predict.right()),int(box1_predict.bottom())),(0,255,255),1)
        cv2.rectangle(image,(int(box2_predict.left()),int(box2_predict.top())),(int(box2_predict.right()),int(box2_predict.bottom())),(0,255,100),1)
        cv2.rectangle(image,(int(box3_predict.left()),int(box3_predict.top())),(int(box3_predict.right()),int(box3_predict.bottom())),(200,0,200),1)
        cv2.imshow('image',image)
        cv2.waitKey(10)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
