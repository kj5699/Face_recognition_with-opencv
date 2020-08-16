#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 00:15:09 2018

@author: jk
"""


import cv2
import numpy as np

from PIL import Image
from facenet_pytorch import MTCNN
import torch

face_cascade=cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
lpb_face_cascade=cv2.CascadeClassifier("cascades/lbpcascade_frontalface.xml")


def frontal_face(classifier,image,scaling_factor):
    image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=classifier.detectMultiScale(image_gray,scaling_factor,5)
    boxes=[]
    
    if len(faces)!=0:
        for x,y,w,h in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2) 
            boxes.append((x,y,w,h))
    
    return image,boxes
    
def face_detect_Facenet(image):
    # check if gpu is available
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn=MTCNN(keep_all=True,device=device)
    boxes,_=mtcnn.detect(image)
    color=(0,255,0)
    if boxes is not None:
        for box in boxes:
            cv2.rectangle(image,(box[0],box[1]),
                          (box[2],box[3]) , color, 2)
    return image,boxes
            


if __name__=="__main__":
    cap=cv2.VideoCapture(0)

    while True:
        ret,image=cap.read()
        if ret:
            #frame,boxes=frontal_face(lpb_face_cascade,image,1.2)
            frame,boxes=face_detect_Facenet(image)
            cv2.imshow("frame",frame)
            k = cv2.waitKey(1)
        
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
        
    
    
    cap.release()
    cv2.destroyAllWindows()
    
    
        
    
