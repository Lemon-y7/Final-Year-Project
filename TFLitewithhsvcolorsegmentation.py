

from __future__ import (
    division, absolute_import, print_function, unicode_literals)
import threading
from networktables import NetworkTables
import tflite_runtime.interpreter as tflite
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

# Capturing video through webcam 
mapping = [False]
lookfor = 'nothing'
red = [False]
blue = [False]
yellow = [False]
green = [False]
black = [False]
objectpicked = 'KitKat'
kernal = np.ones((5, 5), "uint8") 
found = 'none'


cond = threading.Condition()
notified = [False]

def GrayWorld(frame1)::
    frame2 = cv2.cvtColor(frame1, cv2.COLOR_RGB2LAB)
    avg_lab_a = np.average(frame2[:, :, 1])
    avg_lab_b = np.average(frame2[:, :, 2])
    frame2[:, :, 1] = frame2[:, :, 1] - ((avg_lab_a - 128) * (frame2[:, :, 0] / 255.0) * 1.1)
    frame2[:, :, 2] = frame2[:, :, 2] - ((avg_lab_b - 128) * (frame2[:, :, 0] / 255.0) * 1.1)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_LAB2RGB)
    return frame2

def connectionListener(connected, info):
    print(info, '; Connected=%s' % connected)
    with cond:
        notified[0] = True
        cond.notify()
        
NetworkTables.initialize(server='10.43.21.2')
NetworkTables.addConnectionListener(connectionListener, immediateNotify=True)

with cond:
    print("Waiting")
    if not notified[0]:
        cond.wait()

# Insert your processing code here
print("Connected!")

sd = NetworkTables.getTable("SmartDashboard")

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(320,320),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True
        
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Provide the path to the TFLite file, default is models/model.tflite',
                    default='models/model.tflite')
parser.add_argument('--labels', help='Provide the path to the Labels, default is models/labels.txt',
                    default='models/labels.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.98)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='640x480')
                    
args = parser.parse_args()

# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = args.model

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = args.labels

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = float(args.threshold)

resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
import time
print('Loading model...', end='')
start_time = time.time()

# LOAD TFLITE MODEL
interpreter = tflite.Interpreter(model_path=PATH_TO_MODEL_DIR)
# LOAD LABELS
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
print('Running inference for PiCamera')
# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    KitKatall=[]
    Ballall=[]
    Chipsall=[]
    Nissinall=[]
    chipcord=[]
    nissincord=[]
    ballcord=[]
    kitkatcord=[]
 
    # Start timer (for calculating frame rate)
    current_count=0
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame = videostream.read()
    frame = GrayWorld(frame)


  # Acquire frame and resize to expected shape [1xHxWx3]

 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    frame4 = frame.copy()
    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
    
    #Coordinate Reset
    sd.putNumber("KitKatx", 0)
    sd.putNumber("KitKaty", 0)
    sd.putNumber("Nissinx", 0)
    sd.putNumber("Nissiny", 0)
    sd.putNumber("Ballx", 0)
    sd.putNumber("Bally", 0)
    sd.putNumber("Chipsx", 0)
    sd.putNumber("Chipsy", 0)

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > MIN_CONF_THRESH) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            mapping[0] = sd.getBoolean("mapping", True)
            # Example: 'person: 72%'
            if (mapping[0] == False):
            #Coordinate sending
                if (object_name == 'KitKat') :
                    objectpicked = 'KitKat'
                    KitKatx = ((xmax + xmin) / 2) - 320
                    KitKaty = 240 - ((ymax + ymin) / 2)
                    KitKatall = KitKatall+[object_name]
                    kitkatcord= kitkatcord+[[KitKatx,KitKaty]]
                    kkx=(min(kitkatcord,key=lambda x:abs(0)))[0]
                    kky=(min(kitkatcord,key=lambda x:abs(0)))[1]
                    sd.putNumber("KitKatx",kkx)
                    sd.putNumber("KitKaty",kky)
                    label = '%s: %d%% Coord:%d, %d' % (object_name, int(scores[i]*100), KitKatx, KitKaty)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    cv2.putText(frame, 'KitKat Coordinates : ' + str(min(kitkatcord,key=lambda x:abs(0))),(15,85),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,55),2,cv2.LINE_AA)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    
                elif (object_name == 'Ball'):
                    objectpicked = 'Ball'
                    Ballx = ((xmax + xmin) / 2) - 320
                    Bally = 240 - ((ymax + ymin) / 2)
                    Ballall = Ballall+[object_name]
                    ballcord= ballcord+[[Ballx,Bally]]
                    bx=(min(ballcord,key=lambda x:abs(0)))[0]
                    by=(min(ballcord,key=lambda x:abs(0)))[1]
                    sd.putNumber("Ballx",bx )
                    sd.putNumber("Bally",by)
                    label = '%s: %d%% Coord:%d, %d' % (object_name, int(scores[i]*100), Ballx, Bally)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    cv2.putText(frame, 'Ball Coordinates : ' + str(min(ballcord,key=lambda x:abs(0))),(15,105),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,55),2,cv2.LINE_AA)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                elif (object_name == 'Nissin'):
                    objectpicked = 'Nissin'
                    Nissinx = ((xmax + xmin) / 2) - 320
                    Nissiny = 240 - ((ymax + ymin) / 2)
                    Nissinall = Nissinall+[object_name]
                    nissincord= nissincord+[[Nissinx,Nissiny]]
                    nx=(min(nissincord,key=lambda x:abs(0)))[0]
                    ny=(min(nissincord,key=lambda x:abs(0)))[1]
                    sd.putNumber("Nissinx",nx )
                    sd.putNumber("Nissiny",ny )
                    label = '%s: %d%% Coord:%d, %d' % (object_name, int(scores[i]*100), Nissinx, Nissiny)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    cv2.putText(frame, 'Nissin Coordinates : ' + str(min(nissincord,key=lambda x:abs(0))),(15,125),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,55),2,cv2.LINE_AA)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                elif (object_name == 'Chips'):
                    objectpicked = 'Chips'
                    Chipsx = ((xmax + xmin) / 2) - 320
                    Chipsy = 240 - ((ymax + ymin) / 2)
                    Chipsall = Chipsall+[object_name]
                    chipcord= chipcord+[[Chipsx,Chipsy]]
                    cx=(min(chipcord,key=lambda x:abs(0)))[0]
                    cy=(min(chipcord,key=lambda x:abs(0)))[1]
                    sd.putNumber("Chipsx",cx )
                    sd.putNumber("Chipsy",cy)
                    label = '%s: %d%% Coord:%d, %d' % (object_name, int(scores[i]*100), Chipsx, Chipsy)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    cv2.putText(frame, 'Chips Coordinates : ' + str(min(chipcord,key=lambda x:abs(0))),(15,145),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,55),2,cv2.LINE_AA)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                
                current_count+=1                

            elif (mapping[0] == True):
                
                objectpicked = 'KitKat'
                if (objectpicked == 'KitKat'):
                    lookfor = 'RedBox'
                elif (objectpicked == 'Ball'):
                    lookfor = 'BlueBox'
                elif (objectpicked == 'Nissin'):
                    lookfor = 'YellowBox'
                elif (objectpicked == 'Chips'):
                    lookfor = 'GreenBox'
                else:
                    lookfor = 'BlackBox'

        

                if (lookfor == 'RedBox'):
                    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                    red_lower = np.array([114, 155, 43], np.uint8) 
                    red_upper = np.array([148, 255, 80], np.uint8) 
                    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
                    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernal)
                    res_red = cv2.bitwise_and(frame, frame, mask = red_mask)
                    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
                    for pic, contour in enumerate(contours): 
                        area = cv2.contourArea(contour) 
                        if(area > 100):
                            red = [True]
                            found = 'RedBox'
                            x, y, w, h = cv2.boundingRect(contour) 
                            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            #                 imageFrame = cv2.circle(imageFrame,(x+w/2,y+h/2), 10, (0,0,0), -1)
                            cv2.putText(frame, "Red Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
                        

                # Set range for green color and 
                # define mask
                elif (lookfor == 'GreenBox'):
                    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                    green_lower = np.array([40, 35, 0], np.uint8) 
                    green_upper = np.array([82, 255, 131], np.uint8) 
                    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
                    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernal)
                    res_green = cv2.bitwise_and(frame, frame, mask = green_mask)
                # Creating contour to track green color 
                    contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for pic, contour in enumerate(contours): 
                        area = cv2.contourArea(contour) 
                        if(area > 1000):
                            green = [True]
                            found = 'GreenBox'
                            x, y, w, h = cv2.boundingRect(contour) 
                            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(frame, "Green Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
                        
                elif (lookfor == 'BlueBox'):
                    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                    # Set range for blue color and 
                    # define mask 
                    blue_lower = np.array([9, 86, 88], np.uint8) 
                    blue_upper = np.array([22, 172, 139], np.uint8) 
                    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)
                    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernal)
                    res_blue = cv2.bitwise_and(frame, frame, mask = blue_mask)
                    # Creating contour to track blue color 
                    contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for pic, contour in enumerate(contours): 
                        area = cv2.contourArea(contour) 
                        if(area > 1000):
                            blue = [True]
                            found = 'BlueBox'
                            x, y, w, h = cv2.boundingRect(contour) 
                            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            cv2.putText(frame, "Blue Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))
                            
                    
                elif (lookfor == 'BlackBox'):
                    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                    black_lower = np.array([113, 32, 15], np.uint8)
                    black_upper = np.array([169, 150, 30], np.uint8)
                    black_mask = cv2.inRange(hsvFrame, black_lower, black_upper)
                    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernal)
                    res_black = cv2.bitwise_and(frame, frame, mask = black_mask)
                    
                    contours, hierarchy = cv2.findContours(black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for pic, contour in enumerate(contours): 
                        area = cv2.contourArea(contour) 
                        if(area > 1000):
                            black = [True]
                            found = 'BlackBox'
                            x, y, w, h = cv2.boundingRect(contour) 
                            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
                            cv2.putText(frame, "black Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
                    
                        
                elif (lookfor == 'YellowBox'):
                    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                    yellow_lower = np.array([79, 86, 113], np.uint8) 
                    yellow_upper = np.array([110, 170, 162], np.uint8)
                    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper) 
                    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernal)
                    res_yellow = cv2.bitwise_and(frame, frame, mask = yellow_mask)    
                    contours, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for pic, contour in enumerate(contours): 
                        area = cv2.contourArea(contour) 
                        if(area > 1000):
                            yellow = [True]
                            found = 'YellowBox'
                            x, y, w, h = cv2.boundingRect(contour) 
                            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0,234,255), 2)
                            cv2.putText(frame, "Yellow Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,234,255))
                else:
                    red = [False]
                    blue = [False]
                    yellow = [False]
                    green = [False]
                    black = [False]
        
    print(lookfor)
    print(red[0])
    print(blue[0])
    print(yellow[0])
    print(green[0])
    print(black[0])   
    
    cv2.circle(frame,(320,240),5,(255,255,0),cv2.FILLED)
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(15,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,55),2,cv2.LINE_AA)
    cv2.putText (frame,'Total Detection Count : ' + str(current_count),(15,65),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,55),2,cv2.LINE_AA)
            

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object Detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()
print("Done")    
