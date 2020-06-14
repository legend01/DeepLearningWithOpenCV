'''
@Description: 
@version: 
@Author: HLLI8
@Date: 2020-06-14 16:34:38
@LastEditors: HLLI8
@LastEditTime: 2020-06-14 16:35:07
'''
# import the necessary packages
from imutils.video import VideoStream
import imutils
import time
import cv2
# grab a reference to the webcam
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
vs = VideoStream(src="nvarguscamerasrc ! video/x-raw(memory:NVMM), " \
	"width=(int)1920, height=(int)1080,format=(string)NV12, " \
	"framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, " \
	"format=(string)BGRx ! videoconvert ! video/x-raw, " \
	"format=(string)BGR ! appsink").start()
time.sleep(2.0)

# loop over frames
while True:
	# grab the next frame
	frame = vs.read()
	# resize the frame to have a maximum width of 500 pixels
	frame = imutils.resize(frame, width=500)
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# release the video stream and close open windows
vs.stop()
cv2.destroyAllWindows()