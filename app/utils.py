import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as pt
import pickle
from PIL import Image
from glob import glob
from sklearn.decomposition import PCA


haar=cv2.CascadeClassifier("./model/haarcascade_frontalface_default.xml")
mean=pickle.load(open("./model/mean_preprocess.pickle",'rb'))
model_svm=pickle.load(open("./model/ml_model_svm.pickle",'rb'))
model_pca=pickle.load(open("./model/pca_50.pickle",'rb'))

print("model loaded successfully")

gender_pre=["male","female"]
font=cv2.FONT_HERSHEY_SIMPLEX

def ml_model(path,filename,color="bgr"):
    image=cv2.imread(path)
    if color=="bgr":
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    else:
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    faces=haar.detectMultiScale(gray,1.3,5)
    print(faces)
    for x,y,w,h in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
        crop=gray[y:y+h,x:x+w]
        crop=crop/255.0
        if crop.shape[1]>100:
            crop_resize=cv2.resize(crop,(100,100),cv2.INTER_AREA)
        else:
             crop_resize=cv2.resize(crop,(100,100),cv2.INTER_CUBIC)
        pt.imshow(crop_resize)
        crop_reshape=crop_resize.reshape(1,-1)
        crop_mean=crop_reshape-mean
        eigen_image=model_pca.transform(crop_mean)
        results=model_svm.predict_proba(eigen_image)[0]
        predict=results.argmax()
        print(results)
        score=results[predict]
        text="%s : %0.2f"%(gender_pre[predict],score)
        cv2.putText(image,text,(x,y),font,1,(0,255,0),2)
    cv2.imwrite("./static/predict/{}".format(filename),image)

def facedetection(path,filename,color="bgr"):
    image=cv2.imread(path)
    if color=="bgr":
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    else:
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    faces=haar.detectMultiScale(gray,1.3,5)
    print(faces)
    for x,y,w,h in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
        crop=image[y:y+h,x:x+w]
        cv2.imwrite("./static/faces/{}".format(filename),crop)
             