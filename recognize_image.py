from imutils import paths
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

model="face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
protopath="face_detection_model/deploy.prototxt"
embedder="openface_nn4.small2.v1.t7"

detector=cv2.dnn.readNetFromCaffe(protopath, model)
embedder=cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")
recognier =pickle.loads(open("output/recognizer.pickle","rb").read())
le =pickle.loads(open("output/le.pickle","rb").read())

CONFIDENCE=0.5

frame=cv2.imread("sample_images/image1.jpg")

	
frame=imutils.resize(frame,width=600)
h,w=frame.shape[:2]
imageBlob=cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
	1.0,(300,300),(104.0,177.0,123.0),swapRB=False,crop=False)

detector.setInput(imageBlob)
detections=detector.forward()
print(detections.shape)

if len(detections)>0:
	i=np.argmax(detections[0,0,:,2])
	confidence=detections[0,0,i,2]

	if confidence>CONFIDENCE:
		box=detections[0,0,i,3:7]*np.array([w,h,w,h])
		(startX, startY, endX, endY) = box.astype("int")
		face = frame[startY:endY, startX:endX]
		(fH, fW) = face.shape[:2]

		faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
			(96, 96), (0, 0, 0), swapRB=True, crop=False)
		
		embedder.setInput(faceBlob)
		vec = embedder.forward()

		preds=recognier.predict_proba(vec)[0]
		j=np.argmax(preds)
		proba=preds[j]
		name=le.classes_[j]
		text = "{}: {:.2f}%".format(name, proba * 100)
		y = startY - 10 if startY - 10 > 10 else startY
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
	
cv2.imshow("Frame",frame)
cv2.waitKey(0) 


# do a bit of cleanup
cv2.destroyAllWindows()

