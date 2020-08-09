#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 19:14:18 2018

@author: jk
"""
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from face_detection_haar_cascade import frontal_face

face_classifier=cv2.CascadeClassifier("cascades/lbpcascade_frontalface.xml")
 
def create_dataset():
    
    user=str(input("Enter user name:"))
    Id=input('enter your id: ')
    os.makedirs("dataset/s"+str(Id))
    cap=cv2.VideoCapture(0)
    cv2.namedWindow("train")
    sampleNum= 0
    while(True):

        ret, img = cap.read()
        if ret:
            img,boxes=frontal_face(face_classifier,image,1.2)
            if not len(boxes)!=0:
                for box in boxes:
                    x,y,w,h=box
                    k=cv2.waitKey(100)
                    
                    cv2.imshow('frame',img)
                    
                    if k%256 == 32 :
                        img_name = "dataset/s"+str(Id)+"/"+"{}.png".format(sampleNum)
                        cv2.imwrite(img_name, img)
                        print("{} written!".format(img_name))
                        sampleNum += 1
                        
        k = cv2.waitKey(1)

        if k%256 == 27:
        # ESC pressed
            print("Escape hit, closing...")
            break
        elif sampleNum>5:
            break
    
    subjects.subjects.append(user)
    cap.release()
    cv2.destroyAllWindows()
        
if __name__=="__main__":
    create_dataset()























