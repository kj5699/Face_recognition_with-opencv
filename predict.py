#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 11:36:52 2018

@author: jk
"""
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def predict4(test_img):
#make a copy of the image as we don't want to change original image
    img = test_img.copy()
#detect face from the image
    face, rect = detect_face(img)
    if face is None:
        print("no faces found")
        return img
#predict the image using our face recognizer 
    else:
        label= face_recognizer.predict(face)
        
        if (label[1]>50):
            print("faces not found")
            draw_rectangle(img, rect)
            draw_text(img, "Not matched", rect[0], rect[1]-5)
            return img
        else:    
#get name of respective label returned by face recognizer
            label_text = subjects[label[0]]
 
#draw a rectangle around face detected
            draw_rectangle(img, rect)
#draw name of predicted person
            draw_text(img, label_text, rect[0], rect[1]-5)
 
            return img
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
"""def predict4(test_img):
#make a copy of the image as we don't want to change original image
    img = test_img.copy()
#detect face from the image
    face,rect = detect_face(img)
    if face is None:
        print("no faces found")
        return img
#predict the image using our face recognizer 
    else:
        label= face_recognizer.predict(face)
        
        if (label[1]>0.2):
            #print("faces not found")
            create_dataset()
            
            return img
        else:    
#get name of respective label returned by face recognizer
            label_text = subjects[label[0]]
 
#draw a rectangle around face detected
            draw_rectangle(img, rect)
#draw name of predicted person
            draw_text(img, label_text, rect[0], rect[1]-5)
 
            return img"""