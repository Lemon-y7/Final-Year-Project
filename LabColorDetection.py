
#Original Code from https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa3VFOFJMZ2dSRzRIVEcwVS1yTXh3ZnVpR1VpUXxBQ3Jtc0tsZmtWMW1qaXdTTmxCOHF3Mkg1U0EzRGNRYlNBcTgzemFwZHNFN2VwMXJvb2h0clpWNTN2MjRkMzVZMVVkeDdUbXJqZk9ETHliSVp4VFBJMy1wU3dMNnp1OWh4X0FITjBpMXlscWxKMWhuV2phLWdmQQ&q=https%3A%2F%2Fdrive.google.com%2Fdrive%2Ffolders%2F1nU2WJfIErrzsA_BspmShIQmuLHf_w6bn%3Fusp%3Dsharing
#Original Code from https://www.youtube.com/watch?v=6Otgyyv--UU
from __future__ import (
    division, absolute_import, print_function, unicode_literals)
import numpy as np 
import cv2 

kernal = np.ones((5, 5), "uint8") 
# Capturing video through webcam 
webcam = cv2.VideoCapture(0)
red = [False]
blue = [False]
yellow = [False]
green = [False]
black = [False]

objectpicked = 'KitKat'
found = 'none'
pickedup= [True]
# Start a while loop 
while(1): 
    
    # Reading the video from the 
    # webcam in image frames 
    _, imageFrame = webcam.read()
    result = cv2.cvtColor(imageFrame, cv2.COLOR_RGB2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    imageFrame = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
    
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
    labFrame = cv2.cvtColor(imageFrame, cv2.COLOR_RGB2LAB)
    
    if (lookfor == 'redbox'):
        # Set range for red color and 
        # define mask
        red_lower = np.array([0, 0, 0], np.uint8) 
        red_upper = np.array([113, 255, 98], np.uint8) 
        red_mask = cv2.inRange(labFrame, red_lower, red_upper)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernal)
        res_red = cv2.bitwise_and(imageFrame, imageFrame, mask = red_mask)
        contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
        for pic, contour in enumerate(contours): 
            area = cv2.contourArea(contour) 
            if(area > 3000):
                red = [True]
                found = 'redbox'
                x, y, w, h = cv2.boundingRect(contour) 
                imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                imageFrame = cv2.circle(imageFrame,(x+w/2,y+h/2), 10, (0,0,0), -1)
                cv2.putText(imageFrame, "Red Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
            

    # Set range for green color and 
    # define mask
    elif (lookfor == 'greenbox'):
        green_lower = np.array([58, 22, 58], np.uint8) 
        green_upper = np.array([83, 112, 118], np.uint8) 
        green_mask = cv2.inRange(labFrame, green_lower, green_upper)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernal)
        res_green = cv2.bitwise_and(imageFrame, imageFrame, mask = green_mask)
    # Creating contour to track green color 
        contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for pic, contour in enumerate(contours): 
            area = cv2.contourArea(contour) 
            if(area > 3000):
                green = [True]
                found = 'greenbox'
                x, y, w, h = cv2.boundingRect(contour) 
                imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(imageFrame, "Green Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
            
    elif (lookfor == 'bluebox'):
        # Set range for blue color and 
        # define mask 
        blue_lower = np.array([12, 84, 102], np.uint8) 
        blue_upper = np.array([19, 143, 144], np.uint8) 
        blue_mask = cv2.inRange(labFrame, blue_lower, blue_upper)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernal) 
        res_blue = cv2.bitwise_and(imageFrame, imageFrame, mask = blue_mask)
        # Creating contour to track blue color 
        contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours): 
            area = cv2.contourArea(contour) 
            if(area > 3000):
                blue = [True]
                found = 'bluebox'
                x, y, w, h = cv2.boundingRect(contour) 
                imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(imageFrame, "Blue Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))
                   
        
    elif (lookfor == 'blackbox'):
        black_lower = np.array([113, 32, 15], np.uint8)
        black_upper = np.array([169, 150, 30], np.uint8)
        black_mask = cv2.inRange(labFrame, black_lower, black_upper)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernal)
        res_black = cv2.bitwise_and(imageFrame, imageFrame, mask = black_mask)
        
        contours, hierarchy = cv2.findContours(black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours): 
            area = cv2.contourArea(contour) 
            if(area > 3000):
                black = [True]
                found = 'blackbox'
                x, y, w, h = cv2.boundingRect(contour) 
                imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 0), 2)
                cv2.putText(imageFrame, "black Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
        
            
    elif (lookfor == 'yellowbox'):    
        yellow_lower = np.array([95, 67, 156], np.uint8) 
        yellow_upper = np.array([112, 131, 188], np.uint8) 
        yellow_mask = cv2.inRange(labFrame, yellow_lower, yellow_upper) 
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernal) 
        res_yellow = cv2.bitwise_and(imageFrame, imageFrame, mask = yellow_mask)    
        contours, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours): 
            area = cv2.contourArea(contour) 
            if(area > 3000):
                yellow = [True]
                found = 'yellowbox'
                x, y, w, h = cv2.boundingRect(contour) 
                imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0,234,255), 2)
                cv2.putText(imageFrame, "Yellow Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,234,255))
    else:
        red = [False]
        blue = [False]
        yellow = [False]
        green = [False]
        black = [False]
    if (found == lookfor):
        print("Lock On! Stop moving") #stop moving if box is found

        
    else:
        print("Not Found")
        
    print(lookfor)

    print(red[0])
    print(blue[0])
    print(yellow[0])
    print(green[0])
    print(black[0])
    print('\n')
    
    # Program Termination 
    cv2.imshow("LabColorDetection", imageFrame) 
    if cv2.waitKey(10) & 0xFF == ord('q'): 
        cap.release() 
        cv2.destroyAllWindows() 
        break

