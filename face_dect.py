#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 11:39:55 2018

@author: jkk
"""
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def detect_face(img):
#convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
#load OpenCV face detector, I am using LBP which is fast
#there is also a more accurate but slow: Haar classifier
    face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    lpb_face_cascade=cv2.CascadeClassifier("lbpcascade_frontalface.xml")
 
#let's detect multiscale images(some images may be closer to camera than others)
#result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
 
#if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
 
#under the assumption that there will be only one face,
#extract the face area
    
    (x, y, w, h) = faces[0]
 
#return only the face part of the image
    
    return gray[y-10:y+w+10, x-10:x+h+10], faces[0]










