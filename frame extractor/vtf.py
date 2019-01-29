import cv2
import math
import os
import time

def video_to_frames(input_loc, output_loc,video_name):

    count = 0
    cap = cv2.VideoCapture(input_loc)   # capturing the video from the given path
    frameRate = cap.get(5) #frame rate
    print (frameRate)

    while(cap.isOpened()):
        frameId = cap.get(30) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            filename = video_name + "frame%d.jpg" % count;count+=1
            cv2.imwrite(output_loc + filename, frame)
            
    cap.release()
    print ("Done!")
