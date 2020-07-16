from flask import Flask, render_template, request,send_file
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from ml.YOLO_V3 import YOLO_V3
import numpy as np
import hashlib
import types
import cv2
import sys
import os

app = Flask(__name__)
def process_image(path_img):

    name_img = hashlib.md5(os.urandom(32)).hexdigest()
    print(str(name_img))
    
    img = cv2.imread(path_img,cv2.IMREAD_COLOR)
    tam_orig = img.shape[1],img.shape[0]
    img = cv2.resize(img, (864,480))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    yolo = YOLO_V3()
    result = yolo.predict(img)
    print(result)
    box = result[3]
    rect=(int(box[0]),int(box[1]),int(box[2]),int(box[3]))
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, tam_orig)
    cv2.imwrite(str(name_img)+'.jpg', img)
    path_new_img = name_img+'.jpg'
    return path_new_img

@app.route('/upload')
def upload_file():
    return render_template('upload.html')

@app.route('/process_file', methods = ['GET', 'POST'])
def upload_file2():
    #print(request.method)
    if request.method == 'POST':
        print('post')
        f = request.files['file']
        f.save(secure_filename(f.filename))
        path_img = process_image(secure_filename(f.filename))
        print(path_img)
        #send_file()
        return send_file(path_img, mimetype='')#'<img src="img/input.jpg" alt="Minha Figura">'
if __name__ == "__main__":
  port = int(os.environ.get("PORT", 5000))
  app.run(host='0.0.0.0', port=port)
#app.run()