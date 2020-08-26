# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 19:41:16 2020

@author: Jatin
"""
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle




if __name__== "__main__" :
    
    data =pickle.loads(open("output/embeddings.pickle","rb").read())
    le=LabelEncoder()
    labels=le.fit_transform(data["names"])

    recognizer=SVC(C=1.0,kernel='linear',probability=True)
    recognizer.fit(data["embeddings"],labels)

    f = open("output/recognizer.pickle", "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    f = open("output/le.pickle", "wb")
    f.write(pickle.dumps(le))
    f.close()

