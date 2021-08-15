# The TFLite deployment code is from Edje Electronics' with some minor adjustments by armaanpriyadarshan in https://github.com/armaanpriyadarshan/TensorFlow-2-Lite-Object-Detection-on-the-Raspberry-Pi/blob/main/TFLite-PiCamera-od.py 
# Credit for TFLite deployment code goes to his repo: https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi
#Original Code from https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa3VFOFJMZ2dSRzRIVEcwVS1yTXh3ZnVpR1VpUXxBQ3Jtc0tsZmtWMW1qaXdTTmxCOHF3Mkg1U0EzRGNRYlNBcTgzemFwZHNFN2VwMXJvb2h0clpWNTN2MjRkMzVZMVVkeDdUbXJqZk9ETHliSVp4VFBJMy1wU3dMNnp1OWh4X0FITjBpMXlscWxKMWhuV2phLWdmQQ&q=https%3A%2F%2Fdrive.google.com%2Fdrive%2Ffolders%2F1nU2WJfIErrzsA_BspmShIQmuLHf_w6bn%3Fusp%3Dsharing
#Original Code from https://www.youtube.com/watch?v=6Otgyyv--UU
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

red = [False]
blue = [False]
yellow = [False]
green = [False]
black = [False]
objectpicked = 'none'
kernal = np.ones((5, 5), "uint8") 
found = 'none'
pickedup= [True]

cond = threading.Condition()
notified = [False]

def connectionListener(connected, info):
    print(info, '; Connected=%s' % connected)
    with cond:
        notified[0] = True
        cond.notify()
        
NetworkTables.initialize(server='10.12.34.2')
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
    frame1 = videostream.read()
    result = cv2.cvtColor(frame1, cv2.COLOR_RGB2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    frame2 = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame3 = frame2.copy()
    K_undistort = np.array([[475.31982915, 0.0, 318.89987832],[0.0, 477.06381921, 263.86340059],[0.0, 0.0, 1.0]])
    frame = cv2.undistort(frame3, np.array([[475.31982915, 0.0, 318.89987832],[0.0, 477.06381921, 263.86340059],[0.0, 0.0, 1.0]]), np.array([0.04440637, -0.06559753, 0.00170155, 0.0022771, -0.00950175]),
                                newCameraMatrix=K_undistort)    
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
    sd.putNumber("KitKatx", -9999)
    sd.putNumber("KitKaty", -9999)
    sd.putNumber("Nissinx", -9999)
    sd.putNumber("Nissiny", -9999)
    sd.putNumber("Ballx", -9999)
    sd.putNumber("Bally", -9999)
    sd.putNumber("Chipsx", -9999)
    sd.putNumber("Chipsy", -9999)

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > MIN_CONF_THRESH) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
             # Example: 'person: 72%'

            #Coordinate sending
            if (object_name == 'KitKat') :
                objectpicked = 'KitKat'
                KitKatx = ((xmax + xmin) / 2) - 640
                KitKaty = abs(720 - ((ymax + ymin) / 2))
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
                cv2.putText(frame, 'KitKat Coordinates : ' + str(kkmin[0:2]),(15,105),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,55),2,cv2.LINE_AA)
#                 print(min(kitkatcord,key=lambda x:abs(0)))
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            elif (object_name == 'Ball') :
                objectpicked = 'Ball'
                Ballx = ((xmax + xmin) / 2) - 640
                Bally = abs(720 - ((ymax + ymin) / 2))
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
                cv2.putText(frame, 'Ball Coordinates : ' + str(bmin[0:2]),(15,145),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,55),2,cv2.LINE_AA)
#                 print(min(ballcord,key=lambda x:abs(0)))
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            elif (object_name == 'Nissin') :
                objectpicked = 'Nissin'
                Nissinx = ((xmax + xmin) / 2) - 640
                Nissiny = abs(720 - ((ymax + ymin) / 2))
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
                cv2.putText(frame, 'Nissin Coordinates : ' + str(nmin[0:2]),(15,185),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,55),2,cv2.LINE_AA)
#                 print(min(nissincord,key=lambda x:abs(0)))
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            elif (object_name == 'Chips') :
                objectpicked = 'Chips'
                Chipsx = ((xmax + xmin) / 2) - 640
                Chipsy = abs(720 - ((ymax + ymin) / 2))
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
                cv2.putText(frame, 'Chips Coordinates : ' + str(cmin[0:2]),(15,225),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,55),2,cv2.LINE_AA)
#                 print(min(chipcord,key=lambda x:abs(0)))
#                 print((min(chipcord,key=lambda x:abs(0)))[0])
#                 print((min(chipcord,key=lambda x:abs(0)))[1])
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            
            current_count+=1
    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(15,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,55),2,cv2.LINE_AA)
    cv2.putText (frame,'Total Detection Count : ' + str(current_count),(15,65),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,55),2,cv2.LINE_AA)
    
    if (pickedup[0] == True):
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
    else:
        lookfor = 'nothing'
        

    if (lookfor == 'RedBox'):
        labFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        red_lower = np.array([108, 169, 50], np.uint8) 
        red_upper = np.array([128, 226, 82], np.uint8) 
        red_mask = cv2.inRange(labFrame, red_lower, red_upper)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernal)
        res_red = cv2.bitwise_and(frame, frame, mask = red_mask)
        contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
        for pic, contour in enumerate(contours): 
            area = cv2.contourArea(contour) 
            if(area > 1000):
                red = [True]
                found = 'RedBox'
                x, y, w, h = cv2.boundingRect(contour) 
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
#                 imageFrame = cv2.circle(imageFrame,(x+w/2,y+h/2), 10, (0,0,0), -1)
                cv2.putText(frame, "Red Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
            

    # Set range for green color and 
    # define mask
    elif (lookfor == 'GreenBox'):
        labFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        green_lower = np.array([40, 35, 0], np.uint8) 
        green_upper = np.array([82, 255, 131], np.uint8) 
        green_mask = cv2.inRange(labFrame, green_lower, green_upper)
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
        labFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        # Set range for blue color and 
        # define mask 
        blue_lower = np.array([9, 86, 88], np.uint8) 
        blue_upper = np.array([22, 172, 139], np.uint8) 
        blue_mask = cv2.inRange(labFrame, blue_lower, blue_upper)
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
        labFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        black_lower = np.array([113, 32, 15], np.uint8)
        black_upper = np.array([169, 150, 30], np.uint8)
        black_mask = cv2.inRange(labFrame, black_lower, black_upper)
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
        labFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        yellow_lower = np.array([79, 86, 113], np.uint8) 
        yellow_upper = np.array([110, 170, 162], np.uint8)
        yellow_mask = cv2.inRange(labFrame, yellow_lower, yellow_upper) 
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
