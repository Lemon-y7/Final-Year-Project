#This is the vision code done for our FYP project.
#Original code can be found at https://github.com/armaanpriyadarshan/TensorFlow-2-Lite-Object-Detection-on-the-Raspberry-Pi/blob/main/TFLite-PiCamera-od.py
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
import math

mapping = [True]
cond = threading.Condition()
notified = [False]

def connectionListener(connected, info):
    print(info, '; Connected=%s' % connected)
    with cond:
        notified[0] = True
        cond.notify()
        
def GrayWorld(frame1):
    frame2 = cv2.cvtColor(frame1, cv2.COLOR_RGB2LAB)
    avg_lab_a = np.average(frame2[:, :, 1])
    avg_lab_b = np.average(frame2[:, :, 2])
    frame2[:, :, 1] = frame2[:, :, 1] - ((avg_lab_a - 128) * (frame2[:, :, 0] / 255.0) * 1.1)
    frame2[:, :, 2] = frame2[:, :, 2] - ((avg_lab_b - 128) * (frame2[:, :, 0] / 255.0) * 1.1)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_LAB2RGB)
    return frame2
    #frame2 = whitebalance(frame1)

# def simplest_cb(img, percent):
#     out_channels = []
#     channels = cv2.split(img)
#     totalstop = channels[0].shape[0] * channels[0].shape[1] * percent / 200.0
#     for channel in channels:
#         bc = cv2.calcHist([channel], [0], None, [256], (0,256), accumulate=False)
#         lv = np.searchsorted(np.cumsum(bc), totalstop)
#         hv = 255-np.searchsorted(np.cumsum(bc[::-1]), totalstop)
#         lut = np.array([0 if i < lv else (255 if i > hv else round(float(i-lv)/float(hv-lv)*255)) for i in np.arange(0, 256)], dtype="uint8")
#         out_channels.append(cv2.LUT(channel, lut))
#     return cv2.merge(out_channels)
#     # frame2 = simplest_cb(frame1, 1)

def undistortion(frame2):
    mtx = np.array([[466.85669105,   0.0,         333.73352028],
                   [  0.0,         469.6340314,  227.24203907],
                   [  0.0,           0.0,           1.0        ]])
    newcameramtx = np.array([[458.24636841, 0.0, 343.78826758],
                            [0.0, 455.4083252, 224.71192114],
                            [0.0, 0.0,  1.0]])
    dist = np.array([[ 0.04870416, -0.05438429, -0.00202455,  0.00891188, -0.03262199]])
    frame3 = cv2.undistort(frame2, mtx, dist, None, newcameramtx)
    return frame3


NetworkTables.initialize(server='10.19.85.2')
NetworkTables.addConnectionListener(connectionListener, immediateNotify=True)

with cond:
    print("Waiting")
    if not notified[0]:
        cond.wait()
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
                    default=0.5)
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
print("h" + str(height))
print("w"+str(width))
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
    frame1 = videostream.read()
    frame2 = whitebalance(frame1)

  # Acquire frame and resize to expected shape [1xHxWx3]
    frame = undistortion(frame2)
 
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
    sd.putNumber("GreenBoxx", 0)
    sd.putNumber("GreenBoxy", 0)       
    sd.putNumber("YellowBoxx",0)
    sd.putNumber("YellowBoxy",0)
    sd.putNumber("RedBoxx", 0)
    sd.putNumber("RedBoxy", 0)     
    sd.putNumber("BlackBoxx", 0)
    sd.putNumber("BlackBoxy", 0) 
    sd.putNumber("BlueBoxx", 0)
    sd.putNumber("BlueBoxy", 0)  
    sd.putNumber("Binx", 0)
    sd.putNumber("Biny", 0)  

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
                    KitKatx = ((xmax + xmin) / 2) - 320
                    KitKaty = 240 - ((ymax + ymin) / 2)
                    KitKatall = KitKatall+[object_name]
                    KitKatdist = int(math.sqrt(KitKatx**2+KitKaty**2))
                    kitkatcord= kitkatcord+[[KitKatx,KitKaty, KitKatdist]]
                    kkmin = min(kitkatcord,key = lambda x: x[2])
                    kkx=kkmin[0]
                    kky=kkmin[1]
                    kkd=kkmin[2]
                    sd.putNumber("KitKatx",kkx)
                    sd.putNumber("KitKaty",kky)
                    label = '%s: %d%% Coord:%d, %d' % (object_name, int(scores[i]*100), KitKatx, KitKaty)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    cv2.putText(frame, 'KitKat Coordinates : ' + str(kkmin[0:2]),(15,85),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,55),2,cv2.LINE_AA)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    
                elif (object_name == 'Ball'):
                    Ballx = ((xmax + xmin) / 2) - 320
                    Bally = 240 - ((ymax + ymin) / 2)
                    Ballall = Ballall+[object_name]
                    Balldist = int(math.sqrt(Ballx**2+Bally**2))
                    ballcord= ballcord+[[Ballx,Bally, Balldist]]
                    bmin = min(ballcord,key = lambda x: x[2])
                    bx=bmin[0]
                    by=bmin[1]
                    bd=bmin[2]
                    sd.putNumber("Ballx",bx )
                    sd.putNumber("Bally",by)
                    label = '%s: %d%% Coord:%d, %d' % (object_name, int(scores[i]*100), Ballx, Bally)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    cv2.putText(frame, 'Ball Coordinates : ' + str(bmin[0:2]),(15,105),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,55),2,cv2.LINE_AA)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                elif (object_name == 'Nissin'):
                    Nissinx = ((xmax + xmin) / 2) - 320
                    Nissiny = 240 - ((ymax + ymin) / 2)
                    Nissinall = Nissinall+[object_name]
                    Nissindist = int(math.sqrt(Nissinx**2+Nissiny**2))
                    nissincord= nissincord+[[Nissinx,Nissiny,Nissindist]]
                    nmin = min(nissincord,key = lambda x: x[2])
                    nx=nmin[0]
                    ny=nmin[1]
                    nd=nmin[2]
                    sd.putNumber("Nissinx",nx )
                    sd.putNumber("Nissiny",ny )
                    label = '%s: %d%% Coord:%d, %d' % (object_name, int(scores[i]*100), Nissinx, Nissiny)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    cv2.putText(frame, 'Nissin Coordinates : ' + str(nmin[0:2]),(15,125),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,55),2,cv2.LINE_AA)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                elif (object_name == 'Chips') :
                    Chipsx = ((xmax + xmin) / 2) - 320
                    Chipsy = 240 - ((ymax + ymin) / 2)
                    Chipsall = Chipsall+[object_name]
                    Chipsdist = int(math.sqrt(Chipsx**2+ Chipsy**2))
                    chipcord= kitkatcord+[[Chipsx, Chipsy, Chipsdist]]
                    cmin = min(kitkatcord,key = lambda x: x[2])
                    cx=cmin[0]
                    cy=cmin[1]
                    cd=cmin[2]
                    sd.putNumber("Chipsx",cx )
                    sd.putNumber("Chipsy",cy)
                    label = '%s: %d%% Coord:%d, %d' % (object_name, int(scores[i]*100), Chipsx, Chipsy)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    cv2.putText(frame, 'Chips Coordinates : ' + str(cmin[0:2]),(15,145),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,55),2,cv2.LINE_AA)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                
                current_count+=1


            elif (mapping[0] == True):
      
                if (object_name == 'GreenBox'):
                    GreenBoxx = ((xmax + xmin) / 2) - 320
                    GreenBoxy = 240 - ((ymax + ymin) / 2)
                    sd.putNumber("GreenBoxx", GreenBoxx)
                    sd.putNumber("GreenBoxy", GreenBoxy)
                    label = '%s:%d%% x:%d,y:%d' % (object_name, int(scores[i]*100), GreenBoxx, GreenBoxy)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.putText(frame, 'GreenBox Coordinates : ' + str(GreenBoxx) + ',' + str(GreenBoxy),(15,165),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,55),1,cv2.LINE_AA)
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1) # Draw label text
                    cv2.line(frame,(320,240),(int(GreenBoxx)+320,240-int(GreenBoxy)),(255,255,0),2)
                    
                elif (object_name == 'YellowBox'):
                    YellowBoxx = ((xmax + xmin) / 2) - 320
                    YellowBoxy = 240 - ((ymax + ymin) / 2)
                    label = '%s:%d%% x:%d,y:%d' % (object_name, int(scores[i]*100), YellowBoxx, YellowBoxy)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    cv2.putText(frame, 'YellowBox Coordinates : ' + str(YellowBoxx) + ',' + str(YellowBoxy),(15,185),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,55),1,cv2.LINE_AA)
                    sd.putNumber("YellowBoxx", YellowBoxx)
                    sd.putNumber("YellowBoxy", YellowBoxy)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1) # Draw label text
                    cv2.line(frame,(320,240),(int(YellowBoxx)+320,240-int(YellowBoxy)),(255,255,0),2)
                    
                elif (object_name == 'RedBox'):
                    RedBoxx = ((xmax + xmin) / 2) - 320
                    RedBoxy = 240 - ((ymax + ymin) / 2)
                    sd.putNumber("RedBoxx", RedBoxx)
                    sd.putNumber("RedBoxy", RedBoxy)
                    label = '%s:%d%% x:%d,y:%d' % (object_name, int(scores[i]*100), RedBoxx, RedBoxy)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    cv2.putText(frame, 'RedBox Coordinates : ' + str(RedBoxx) + ',' + str(RedBoxy),(15,205),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,55),1,cv2.LINE_AA)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1) # Draw label text
                    cv2.line(frame,(320,240),(int(RedBoxx)+320,240-int(RedBoxy)),(255,255,0),2)
                                        
                elif (object_name == 'BlackBox'):
                    BlackBoxx = ((xmax + xmin) / 2) - 320
                    BlackBoxy = 240 - ((ymax + ymin) / 2)
                    sd.putNumber("BlackBoxx", BlackBoxx)
                    sd.putNumber("BlackBoxy", BlackBoxy)                    
                    label = '%s:%d%% x:%d,y:%d' % (object_name, int(scores[i]*100), BlackBoxx, BlackBoxy)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    cv2.putText(frame, 'BlackBox Coordinates : ' + str(BlackBoxx) + ',' + str(BlackBoxy),(15,225),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,55),1,cv2.LINE_AA)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1) # Draw label text
                    cv2.line(frame,(320,240),(int(BlackBoxx)+320,240-int(BlackBoxy)),(255,255,0),2)
                                        
                elif (object_name == 'BlueBox'):
                    BlueBoxx = ((xmax + xmin) / 2) - 320
                    BlueBoxy = 240 - ((ymax + ymin) / 2)
                    sd.putNumber("BlueBoxx", BlueBoxx)
                    sd.putNumber("BlueBoxy",  BlueBoxy)
                    label = '%s:%d%% x:%d,y:%d' % (object_name, int(scores[i]*100), BlueBoxx, BlueBoxy) 
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    cv2.putText(frame, 'BlueBox Coordinates : ' + str(BlueBoxx) + ',' + str(BlueBoxy),(15,245),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,55),1,cv2.LINE_AA)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1) # Draw label text
                    cv2.line(frame,(320,240),(int(BlueBoxx)+320,240-int(BlueBoxy)),(255,255,0),2)

                elif (object_name == 'Bin'):
                    Binx = ((xmax + xmin) / 2) - 320
                    Biny = 240 - ((ymax + ymin) / 2)
                    sd.putNumber("Binx", Binx)
                    sd.putNumber("Biny", Biny)
                    label = '%s:%d%% x:%d,y:%d' % (object_name, int(scores[i]*100), Binx, Biny) 
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    cv2.putText(frame, 'Bin Coordinates : ' + str(Binx) + ',' + str(Biny),(15,265),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,55),1,cv2.LINE_AA)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1) # Draw label text
                    cv2.line(frame,(320,240),(int(Binx)+320,240-int(Biny)),(255,255,0),2)
                
                current_count+=1
                

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



