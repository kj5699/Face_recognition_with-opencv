#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 19:14:18 2018

@author: jk
"""
import cv2
import os
import numpy as np
from face_detection import frontal_face, face_detect_Facenet
import subjects
face_classifier=cv2.CascadeClassifier("cascades/lbpcascade_frontalface.xml")
import time
def create_dataset():
    
    user=str(input("Enter user name:"))
    Id=len(os.listdir("dataset"))+1
    
    dir_name="dataset/s"+str(Id)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
    cap=cv2.VideoCapture(0)
    
    sampleNum= 0
    while(True):

        ret, img = cap.read()
        if ret:
            #img,boxes=frontal_face(face_classifier,image,1.2)
            img,boxes=face_detect_Facenet(img)
            cv2.imshow('frame',img)
            if not len(boxes)!=0:
                print("Taking Pic Get Ready")
                g=cv2.waitKey(100)
                if g%256 ==32:
                
                    img_name = "dataset/s"+str(Id)+"/"+"{}.png".format(sampleNum)
                    cv2.imwrite(img_name, img)
                    print("{} written!".format(img_name))
                    sampleNum += 1
                        
        k = cv2.waitKey(1)

        if k%256 == 27:
        # ESC pressed
            print("Escape hit, closing...")
            break
        elif sampleNum>10:
            break
    
    subjects.subjects[Id]=str(user)
    cap.release()
    cv2.destroyAllWindows()
        
if __name__=="__main__":
    create_dataset()























