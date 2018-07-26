#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:22:34 2018

@author: jk
"""
import numpy as np
import cv2
import imutils
import math
top,bottom,left,right=10,350,600,300



lower_skin=np.array([0,15,30])
upper_skin=np.array([22,255,255])
cam=cv2.VideoCapture(0)

while True:
    ret,frame=cam.read()
    
    if frame is None:
        break
    frame= imutils.resize(frame,width=600)
    frame = cv2.flip(frame, 1)
    clone=frame.copy()
    roi=frame[top:bottom,right:left]
    
    blured=cv2.GaussianBlur(roi,(11,11),0)
    hsv=cv2.cvtColor(blured,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,lower_skin,upper_skin)
    mask=cv2.erode(mask,None,iterations=2)
    mask=cv2.dilate(mask,None,iterations=2)
    
    _,cnts,h=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if cnts is not None:
        segmented_hand=max(cnts,key=cv2.contourArea)
        hull=cv2.convexHull(segmented_hand)
        cv2.drawContours(clone,[hull+(right,top)],-1,(0,0,255),1)
        areacnt=cv2.contourArea(segmented_hand)
        areahull=cv2.contourArea(hull)
        epsilon=0.0005*cv2.arcLength(segmented_hand,True)
        approx=cv2.approxPolyDP(segmented_hand,epsilon,True)
        arearatio=((areahull-areacnt)/areacnt)*100
        hull=cv2.convexHull(approx,returnPoints=False)
        defects=cv2.convexityDefects(approx,hull)
        
        
        cv2.drawContours(clone,[segmented_hand+(right,top)],-1,(0,255,0),4)
        
        l=0
        
            
        for i in range(defects.shape[0]):
            s,e,f,d=defects[i,0]
            start=tuple(approx[s][0])
            end=tuple(approx[e][0])
            far=tuple(approx[f][0])
            pt=(100,180)
            
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
            if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                l += 1
                cv2.circle(clone, tuple([sum(x) for x in zip(far,(right,top))]), 8, [211, 84, 0], -1)
    else:
        continue;
    cv2.rectangle(clone,(left,top),(right,bottom),(0,0,255),4)
    font=cv2.FONT_HERSHEY_COMPLEX
    #cv2.putText(clone,l,(0,50),font,2,(255,255,50),3,cv2.LINE_AA)
    print(l)
    cv2.imshow("frame",clone)
    key=cv2.waitKey(1)&0xFF

    if key==ord("q"):
        break


    
cam.release()  
    
cv2.destroyAllWindows()