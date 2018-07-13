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
import subjects
 
def create_dataset():
    
    user=str(input("Enter user name:"))
    Id=input('enter your id: ')
    os.makedirs("dataset/s"+str(Id))
    cap=cv2.VideoCapture(0)
    cv2.namedWindow("train")
    sampleNum= 0
    
    while(True):
        ret, img = cap.read()
        
 
        faces=detect_face_for_data(img)
        if len(faces)==0:
            print("no faces detected")
            cv2.imshow('frame',img)
            continue;
        
        else:
            for (x,y,w,h) in faces:
                #####cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                """sampleNum=sampleNum+1
                cv2.imwrite("datatet/"+user+"."+Id+'.'+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
                cv2.imshow('frame',img)"""
                
                k=cv2.waitKey(100)
                cv2.imshow('frame',img)
                if k%256 == 32:
                    img_name = "dataset/s"+str(Id)+"/"+"{}.png".format(sampleNum)
                    cv2.imwrite(img_name, img)
                    print("{} written!".format(img_name))
                    
                    sampleNum += 1
        k = cv2.waitKey(1)

        if k%256 == 27:
        # ESC pressed
            print("Escape hit, closing...")
            break
    # break if the sample number is morethan 20
        elif sampleNum>50:
            break
    subjects.subjects.append(user)
    cap.release()
    cv2.destroyAllWindows()
    

def detect_face_for_data(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    lpb_face_cascade=cv2.CascadeClassifier("lbpcascade_frontalface.xml")
 

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
 

    """if (len(faces) == 0):
        return None"""
    return faces
























