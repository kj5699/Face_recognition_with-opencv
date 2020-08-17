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


if __name__== "__main__" :

    faces,labels=prepare_training_data("dataset")
    for face in faces:
        print(face.shape)
    labels=np.array(labels).reshape(len(labels),1)
    face_recognizer = cv2.face.LBPHFaceRecognizer_create() #create our LBPH face recognizer 
    face_recognizer.train(faces,labels)
    face_recognizer.write('train.yml')