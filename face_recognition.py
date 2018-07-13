#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 10:51:14 2018

@author: jk
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

subjects = ["", "Ramiz Raja", "Elvis Presley"]
def detect_face(img):
#convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#load OpenCV face detector, I am using LBP which is fast


#there is also a more accurate but slow: Haar classifier
    face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    lpb_face_cascade=cv2.CascadeClassifier("lbpcascade_frontalface.xml")
 
#let's detect multiscale images(some images may be closer to camera than others)
#result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
 
#if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
 
#under the assumption that there will be only one face,
#extract the face area
    
    (x, y, w, h) = faces[0]
 
#return only the face part of the image
    
    return gray[y:y+w, x:x+h], faces[0]






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
            face, rect = detect_face(image)
#for the purpose of this tutorial
#we will ignore faces that are not detected
            if face is not None:
#add face to list of faces
                faces.append(face)
#add label for this face
            labels.append(label)
 
            cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
 
    return faces, labels

print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")
#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))





#create our LBPH face recognizer 
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
 
#or use EigenFaceRecognizer by imgreplacing above line with 
#face_recognizer = cv2.face.createEigenFaceRecognizer()
 
#or use FisherFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.createFisherFaceRecognizer()



face_recognizer.train(faces, np.array(labels))


def draw_rectangle(img, rect):
 (x, y, w, h) = rect
 cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
 #function to draw text on give image starting from
#passed (x, y) coordinates. 
def draw_text(img, text, x, y):
 cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
 
 
 
 









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
        
        if (label[1]>0.2):
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







print("Predicting images...")
 
#load test images
test_img1 = cv2.imread("123.jpg")
test_img2 = cv2.imread("test-data/test2.jpg")
 
#perform a prediction
predicted_img1 = predict4(test_img1)
predicted_img2 = predict4(test_img2)
print("Prediction complete")
 
#display both images
cv2.imshow(subjects[1], predicted_img1)
cv2.imshow(subjects[2], predicted_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()









