# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 19:41:16 2020

@author: Jatin
"""
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from preparedata import prepare_training_data

face_classifier=cv2.CascadeClassifier("cascades/lbpcascade_frontalface.xml")
if __name__== "__main__" :

    faces,labels=prepare_training_data("dataset")
    
    face_classifier=cv2.CascadeClassifier("cascades/lbpcascade_frontalface.xml")
    face_recognizer = cv2.face.LBPHFaceRecognizer_create() #create our LBPH face recognizer 
    face_recognizer.train(faces, np.array(labels))
    face_recognizer.save('train.yml')