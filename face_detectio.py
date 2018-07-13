#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 00:15:09 2018

@author: jk
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
lpb_face_cascade=cv2.CascadeClassifier("lbpcascade_frontalface.xml")

def frontal_face(classifier,image,scaling_factor):
    

    image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    faces=classifier.detectMultiScale(image_gray,scaling_factor,5)
    
    for x,y,w,h in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
        
    return image
    

cap=cv2.VideoCapture(0)
##imread("opencv_frame_1.png")
#cv2.imshow("frame0",frontal_face(face_cascade,img,1.2))
while True:
    ret,image=cap.read()
    cv2.imshow("frame",frontal_face(lpb_face_cascade,image,1.2))
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break


cap.release()
cv2.destroyAllWindows()
    
    
        
    
