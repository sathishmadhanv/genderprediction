from flask import url_for,redirect,render_template,request
import os
from PIL import Image
from app import utils
import glob

def base():
    return render_template('base.html') 
def index():
    return render_template('index.html')
def faceapp():
    return render_template('faceapp.html')
def wid(path):
    img=Image.open(path)
    size=img.size
    aspect=size[0]/size[1]
    w=300*aspect
    return  w

U_f='static/uploads/'
def gender():
    if request.method=="POST":
        f=request.files['image']
        path=os.path.join(U_f+f.filename)
        f.save(path)
        w=wid(path)
        utils.ml_model(path,f.filename,color='bgr')
        print("file saved sucessfully")
        return render_template("gender.html",Upload=True,fname=f.filename,w=w)

    return render_template("gender.html",Upload=False,w=300)

def face():
    files=glob.glob('static/faces/*')
    for i in files:
        os.remove(i)
    if request.method=="POST":
        f=request.files['image']
        path=os.path.join(U_f+f.filename)
        f.save(path)
        w=wid(path)
        utils.facedetection(path,f.filename,color='bgr')
        print("file saved sucessfully")
        return render_template("faceapp.html",Upload=True,fname=f.filename,w=w)

    return render_template("faceapp.html",Upload=False,w=300)