import cv2
import sys
import types
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
import os
class YOLO_V3():
	def __init__(self,weights_path = os.path.join(os.getcwd(), 'ml/yolov3.weights'),cfg_path=os.path.join(os.getcwd(), 'ml/yolov3.cfg')):
		print(os.path.join(os.getcwd(), 'ml/yolov3.weights'))
		self.classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus",
						"train", "truck", "boat", "traffic light", "fire hydrant",
						"stop sign", "parking meter", "bench", "bird", "cat", "dog",
						"horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
						"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
						"skis", "snowboard", "sports ball", "kite", "baseball bat", 
						"baseball glove", "skateboard", "surfboard", "tennis racket",
						"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
						"banana", "apple", "sandwich", "orange", "broccoli", "carrot",
						"hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
						"bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
						"keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
						"refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
						"hair drier", "toothbrush"]
		self.weights_path    = weights_path
		self.cfg_path 	     = cfg_path
		self.conf_threshold  = 0.5
		self.nms_threshold   = 0.4
	def init_model(self,img):
		model = cv2.dnn.readNet(self.weights_path, self.cfg_path)
		scale = 1./255
		dims = img.shape
		blob = cv2.dnn.blobFromImage(img, scale, (dims[1], dims[0]), (0,0,0), True, crop=False)
		model.setInput(blob)
		outs = model.forward(self.get_output_layers(model))
		return [model,outs]
	def get_output_layers(self,model):
	    layer_names = model.getLayerNames()
	    output_layers = [layer_names[i[0] - 1] for i in model.getUnconnectedOutLayers()]
	    return output_layers

	def draw_bounding_box(self,img, class_id, confidence, x, y, x_plus_w, y_plus_h):
	    label = str(self.classes[class_id])
	    color1 = np.array([0.0,0.0,255.]) # red
	    color2 = np.array([0.0,255.0,255.0])# Other yellow
	    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color1, 2)
	    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color2, 2)
	    return img
		
	def predict(self,img):
		dims = img.shape
		Width=dims[1]
		Height=dims[0]
		aux = self.init_model(img)
		outs  = aux[1]
		model  = aux[0]
		#print(outs)

		# initialization
		class_ids = []
		confidences = []
		boxes = []

		for out in outs:
		    for detection in out:
		        scores = detection[5:]
		        class_id = np.argmax(scores)
		        confidence = scores[class_id]
		        if confidence > 0.8:
		            center_x = int(detection[0] * Width)
		            center_y = int(detection[1] * Height)
		            w = int(detection[2] * Width)
		            h = int(detection[3] * Height)
		            x = center_x - w / 2
		            y = center_y - h / 2
		            class_ids.append(class_id)
		            confidences.append(float(confidence))
		            boxes.append([x, y, w, h])
		indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
		return [class_ids[0],self.classes[class_ids[0]],confidences[0],boxes[0],indices[0]]  
