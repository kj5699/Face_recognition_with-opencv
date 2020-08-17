#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 11:38:05 2018

@author: jk"""
import cv2
import os
import numpy as np

from face_detection import frontal_face, face_detect_Facenet

face_classifier=cv2.CascadeClassifier("cascades/lbpcascade_frontalface.xml")


def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []  #list to hold all subject faces
    labels = [] #list to hold labels for all subjects
 
    for dir_name in dirs: 
        if not dir_name.startswith("s"):
            continue; 
        label = int(dir_name.replace("s", "")) #removing letter 's' from dir_name will give us label

        subject_dir_path = data_folder_path + "/" + dir_name 
        subject_images_names = os.listdir(subject_dir_path) #get the images names that are inside the given subject directory
 
        #------STEP-3--------
        #go through each image name, read image, 
        #detect face and add face to list of faces
        for image_name in subject_images_names:

            if image_name.startswith("."):
                continue;
            image_path = subject_dir_path + "/" + image_name #sample image path = training-data/s1/1.pgm
            image = cv2.imread(image_path)
            cv2.imshow("Training on image...", image)
            cv2.waitKey(100)
            
            # detect face
            #img,boxes=frontal_face(face_classifier,image,1.2)
            img,boxes=face_detect_Facenet(image)
            if len(boxes)!=0:
                for box in boxes:
                    x,y,w,h=box
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    face = gray[int(y)-10:int(y+w)+10, int(x)-10:int(x+h)+10]
                    faces.append(np.array(face,dtype='uint8'))
                    labels.append(int(label))
 
            cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
 
    return faces, labels

"""
print("Preparing data...")
faces, labels = prepare_training_data("dataset")
print("Data prepared")
#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))
"""
