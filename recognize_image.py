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
from facenet_pytorch import MTCNN
import torch
from PIL import Image,ImageDraw
from torch.autograd import Variable
from torchvision import datasets ,transforms, models







def predict_image(image):
    image_tensor=transform(image).float()
    image_tensor=image_tensor.unsqueeze_(0)
    input= Variable(image_tensor).to(device)
    output=model(input)
    
    index=output.data.cpu().numpy().argmax()
    return index

def predict_gender(face):
	img = Image.fromarray(face)
	img=check_transforms(img)
	index=predict_image(img)
	label=str(classes[index])

	return label




if  __name__=="__main__":

	# set up gpu
	device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(device)

	#Load Model for gender
	model=torch.load('classifymodel.pth')
	model.eval()

	# face detector model
	modelpath="face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
	protopath="face_detection_model/deploy.prototxt"
	
	detector=cv2.dnn.readNetFromCaffe(protopath, modelpath)

	# load embedder
	embedder="openface_nn4.small2.v1.t7"
	embedder=cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

	# load recognizer and label encoder
	recognier =pickle.loads(open("output/recognizer.pickle","rb").read())
	le =pickle.loads(open("output/le.pickle","rb").read())

	CONFIDENCE=0.5
	classes=['female','male']

	# tranfor for gender recog model
	transform=transforms.Compose([transforms.Resize((80,80)),transforms.ToTensor(),])
	check_transforms=transforms.Compose([transforms.Resize((80,80)),transforms.ToTensor(),
	                                     transforms.ToPILImage(),])



	# input image
	image=cv2.imread("Sample Images/Image1.jpg")
	image=imutils.resize(image,width=600)
	frame=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		
	h,w=frame.shape[:2]
	imageBlob=cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		1.0,(300,300),(104.0,177.0,123.0),swapRB=False,crop=False)

	
	# Face Detection
	detector.setInput(imageBlob)
	detections=detector.forward()
	print(len(detections))


	for i in range(0, detections.shape[2]):
		
		confidence=detections[0,0,i,2]
		if confidence>CONFIDENCE:
			box=detections[0,0,i,3:7]*np.array([w,h,w,h])
			(startX, startY, endX, endY) = box.astype("int")
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]
			if  fW < 20 or fH < 20:
				continue
			# predict gender
			gender=predict_gender(face)

			# Face Recognition
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			preds=recognier.predict_proba(vec)[0]
			j=np.argmax(preds)
			proba=preds[j]
			name=le.classes_[j]
			
			# write Labels in image
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY
		
			cv2.rectangle(image, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(image, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

			cv2.putText(image, gender, (startX,  y - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)



	cv2.imshow("Frame",image)
	cv2.imwrite("Output3.png",image)
	cv2.waitKey(0) 
	# do a bit of cleanup
	cv2.destroyAllWindows()

