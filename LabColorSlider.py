import cv2
import numpy as np
import sys
import tkinter as tk
from tkinter import filedialog
webcam = cv2.VideoCapture(0)

def getImage():

  ftypes = [
    ("PNG", "*.png;*.PNG"), 
    ("JPG", "*.jpg;*.JPG;*.JPEG"),
    ("GIF", "*.gif;*.GIF"),
    ("All files", "*.*")
  ]
  root = tk.Tk()
  root.withdraw() #HIDE THE TKINTER GUI
  file_path = filedialog.askopenfilename(filetypes = ftypes)
  root.update()

  return file_path

def nothing(x):
    pass


_, src = webcam.read()
frame2 = cv.cvtColor(src, cv.COLOR_RGB2LAB)
# To use grayworld
avg_lab_a = np.average(frame2[:, :, 1])
avg_lab_b = np.average(frame2[:, :, 2])
frame2[:, :, 1] = frame2[:, :, 1] - ((avg_lab_a - 128) * (frame2[:, :, 0] / 255.0) * 1.1)
frame2[:, :, 2] = frame2[:, :, 2] - ((avg_lab_b - 128) * (frame2[:, :, 0] / 255.0) * 1.1)
src = cv.cvtColor(frame2, cv.COLOR_LAB2RGB)
cv2.imshow("Original image", src)
imHSV = cv2.cvtColor(src, cv2.COLOR_RGB2LAB)

##-----------------------------
#Create Trackbar
cv2.namedWindow("Range LAB")
cv2.resizeWindow("Range LAB", 500, 350)

cv2.createTrackbar('Low L', "Range LAB" , 0, 255, nothing)
cv2.createTrackbar('Low A', "Range LAB" , 0, 255, nothing)
cv2.createTrackbar('Low B', "Range LAB" , 0, 255, nothing)

cv2.createTrackbar('High L', "Range LAB" , 0, 255, nothing)
cv2.createTrackbar('High A', "Range LAB" , 0, 255, nothing)
cv2.createTrackbar('High B', "Range LAB" , 0, 255, nothing)


##-----------------------------
#Perform Color Thresholding from Trachbar Position
while(1):
   
    low_L = cv2.getTrackbarPos('Low L',"Range LAB")
    low_A = cv2.getTrackbarPos('Low A',"Range LAB")
    low_B = cv2.getTrackbarPos('Low B',"Range LAB")
   
    high_L = cv2.getTrackbarPos('High L',"Range LAB")
    high_A = cv2.getTrackbarPos('High A',"Range LAB")
    high_B = cv2.getTrackbarPos('High B',"Range LAB")

    low_thresh = np.array([low_L, low_A, low_B])
    high_thresh = np.array([high_L, high_A, high_B])

    # print("low_thresh : ", low_thresh) #printing the array
    # print("high_thresh: ", high_thresh) #printing the array

    imThreshold = cv2.inRange(imHSV, low_thresh, high_thresh)
    cv2.imshow("Thresholded Image", imThreshold)

    imColorThreshold = cv2.bitwise_and(src, src, mask=imThreshold)
    cv2.imshow("Colour Threshold", imColorThreshold)



    if cv2.waitKey(10) & 0xFF == ord('q'): 
        cap.release() 
        cv2.destroyAllWindows() 
        break
