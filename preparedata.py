#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 11:38:05 2018

@author: jk"""
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

import face_dect


def prepare_training_data(data_folder_path):
#get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
 #list to hold all subject faces
    faces = []
#list to hold labels for all subjects
    labels = []
 #let's go through each directory and read images within it
    for dir_name in dirs:
 #our subject directories start with letter 's' so
#ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue;
#extract label number of subject from dir_name
#format of dir name = slabel
#, so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))
 #build path of directory containing images for current subject subject
#sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name
 #get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)
 
#------STEP-3--------
#go through each image name, read image, 
#detect face and add face to list of faces
        for image_name in subject_images_names:
 
#ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;
 
#build image path
#sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name
 
#read image
            image = cv2.imread(image_path)
 
#display an image window to show the image 
            cv2.imshow("Training on image...", image)
            cv2.waitKey(100)
 #detect face
            face, rect = face_dect.detect_face(image)
#for the purpose of this tutorial
#we will ignore faces that are not detected
            #if face is not None:
#add face to list of faces
            faces.append(face)
#add label for this face
            labels.append(label)
 
            cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
 
    return faces, labels

print("Preparing data...")
faces, labels = prepare_training_data("dataset")
print("Data prepared")
#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))
