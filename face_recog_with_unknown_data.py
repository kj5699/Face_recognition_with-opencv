#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 20:19:41 2018

@author: jk
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


subjects=[".","jatin","ankit","sahil"]


    
    



import createdata




import preparedata
faces,labels=preparedata.prepare_training_data("dataset")













#create our LBPH face recognizer 
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
 
#train the recognizer
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
    face, rect = face_dect.detect_face(img)
    if face is None:
        print("no faces found")
        return img
#predict the image using our face recognizer 
    else:
        label= face_recognizer.predict(face)
        
        if (label[1]>50):
            print("faces not found")
            #createdata.create_dataset()
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













import face_dect
print("Predicting images...")
 
#load test images
"""test_img1 = cv2.imread("dataset/s1/4.png",1)
#test_img1=cv2.resize(test_img1,None,fx=0.2,fy=0.2,interpolation=cv2.INTER_CUBIC)
face,rect=face_dect.detect_face(test_img1)
predicted_img = predict4(test_img1)
cv2.imshow("frame", test_img1)



cv2.imshow("frame1", face)


cv2.imshow("test", predicted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

#test_img2 = cv2.imread("test-data/test2.jpg")




cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    #cv2.imshow("frame", frame)
    predicted_img = predict4(frame)
    cv2.imshow("test", predicted_img)
    if not ret:
        break
    
    
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
   


 

print("Prediction complete")
 

cam.release()
cv2.destroyAllWindows()






#or use EigenFaceRecognizer by imgreplacing above line with 
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
 



#or use FisherFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
