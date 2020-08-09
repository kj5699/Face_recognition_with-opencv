#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 20:19:41 2018

@author: jk
"""

import cv2
import numpy as np
from face_detection_haar_cascade import frontal_face
from preparedata import prepare_training_data



def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
 

def predict(test_img):
    img = test_img.copy()
    #detect face from the image
    #face, rect = face_dect.detect_face(img)
    img,boxes=frontal_face(face_classifier,img)
    
    if len(boxes)!=0 :
        print("no faces found")
        return img
    else:
        for box in boxes:
            x,y,w,h=box
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = gray[y-10:y+w+10, x-10:x+h+10]
            label= face_recognizer.predict(face) #predict the image using our face recognizer 
            if (label[1]>50):
                print("faces not found")
                draw_rectangle(img, box)
                draw_text(img, "Not matched", box[0], box[1]-10)
                return img
            
            else:    
                label_text = subjects[label[0]]
                draw_rectangle(img, box)
                draw_text(img, label_text, box[0], box[1]-5)
     
                return img
    


if __name__=="__main__":
    
    subjects=[".","jatin","ankit","sahil"]
    
    faces,labels=prepare_training_data("dataset")
    
    face_classifier=cv2.CascadeClassifier("cascades/lbpcascade_frontalface.xml")
    face_recognizer = cv2.face.LBPHFaceRecognizer_create() #create our LBPH face recognizer 
    face_recognizer.train(faces, np.array(labels)) #train the recognizer
    
    
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    img_counter = 0
    
    while True:
        ret, frame = cam.read()
        if ret:
            predicted_img = predict(frame)
            cv2.imshow("test", predicted_img)
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
