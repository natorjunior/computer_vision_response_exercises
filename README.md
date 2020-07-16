# Solutions for computer vision exercises
### Table of Contents
* 1 - [Optical flow e tracking](#1\)-Optical-flow-e-tracking)
* 2 - [Image stitching](#2\)-Image-stitching)
* 3 - [Model Deployment (Web/Docker)](#3\)-Model-Deployment-\(Web/Docker\))
* 4 - [Object detection com deep learning](#4\)-Object-detection-com-deep-learning)
* 5 - [Context segmentation com deep learning (Avançado)](#5\)-Context-segmentation-com-deep-learning-\(Avançado\))
###### Criando o ambiente de execução do jupyter-notebook:
```bash
#faça o clone do repositório.
$ git clone https://github.com/natorjunior/computer_vision_response_exercises.git
#Entre no diretório 
$ cd computer_vision_response_exercises/
#Faça o build do Dockerfile
$ docker build -t jupyter_notebook:latest .
#Faça a execução do container 
$ docker run -d -p 8888:8888 web_aplication_flask:latest
```


## 1\) Optical flow e tracking
Selecionar uma área de um video e realizar o tracking utilizando Optical Flow. Desenhe o vetor resultante entre as localizações das features.

Utilizar imagens da câmera do dispositivo (notebook ou celular) local, caso não tenha acesso, baixar um vídeo de exemplo e anexar no resultado.
- parse:

<a href="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/optical_flow/output/opticalflow_dense.png"><img src="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/optical_flow/output/opticalflow_dense..png"></a>

- Dense:

<a href="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/optical_flow/output/opticalflow_sparse.jpg"><img src="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/optical_flow/output/opticalflow_sparse.jpg" width=320></a>
  
 
###### Install dependences:

```sh
$ pip install opencv-contrib-python
$ pip install numpy
```
Usage libs
```python
import matplotlib.pyplot as plt
import numpy as np
import cv2
```
Function for calcl optical flow 
```python
cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
```
#### class Optical_flow()
methods
```python
#função para definir os parametros inicias 
def __init__(self):
    pass
#função para calcular o fluxo óptico com o método de Lucas-Kanade, via OpenCv 
def calc_lucas_kened(self,old_gray,frame,p0): #Recebe o frame anterior e o frame atual
    return img #retorna a imagem já calculada 
```
## 2) Image stitching
### Description Solutions
Foram implementadas duas metodologias, uma nativa do OpenCv e outra usando os módulos separados do OpenCv.
Métodos
```python
#Classe com a função via opencv
class image_stitching():
# classe contendo todas as funções para image_stitching
class image_stitching_alternative(): 
# Função para calcular o SIFT via opencv (cv2.xfeatures2d.SIFT_create())
def sift(self,images_data): 
#Calcula os pontos de matcher entre duas imagens. Usa cv2.BFMatcher().knnMatch()
def calc_matcher(self,des1_1, des1_2):#
#Retorna uma imagem contendo as ligações entres os pontos de matches.
def img_matches(self,img1,img2,kp1_1,kp1_2,good):
#faz o join das imagens
def join_matches(self,good,kp1_1,kp1_2,img1):
# Retorna uma imagem com o join das duas imagens 
def img_join(self,img1):
#Faz o corte da imagem de saída
def trim(self,frame):
```


  

Utilizar descritores de imagem, como SURF, SIFT ou ORB para identificar descritores similares entre imagens e conecta-las, gerando uma única imagem.

  

Exemplo de input:

  

<a href="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image1.jpg"><img src="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image1.jpg" align="left" width="128"></a>

  

<a href="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image2.jpg"><img src="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image2.jpg" align="left" width="128"></a>

  

<a href="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image3.jpg"><img src="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image3.jpg" align="left" width="128"></a>

  

<a href="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image4.jpg"><img src="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image4.jpg" align="left" width="128"></a>

  

<a href="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image5.jpg"><img src="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image5.jpg" width="128"></a>

  

Exemplo output:

  

<a href="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/output/output.png"><img src="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/output/output.png" width="512"></a>

## 3\) Model Deployment \(Web/Docker\)
Description Solutions
O algoritmo usa o grabcut do opencv para realizar a retirada do background da imagem, para pegar a área de interesse é ultilizada uma rede de detecção de objetos, a YOLOV3, a mesma seleciona o primeiro objeto que ela achar na imagem e retorna o boundbox, que é passado para a função grabcut do opencv que realizará o processamento e retornará a imagem processada. 
#### Exemplos de execução 
- via browser
    - <img src="3%20-%20Model%20Deployment%20(WebDocker)/img/result_browser.PNG" width=320>
    - <img src="3%20-%20Model%20Deployment%20(WebDocker)/img/result_browser2.PNG" width=320>
- via Postman
    - <img src="3%20-%20Model%20Deployment%20(WebDocker)/img/result_postman.PNG" width=320>
    - <img src="3%20-%20Model%20Deployment%20(WebDocker)/img/result_postman2.PNG" width=320>

###### Usando o docker:
```bash
#faça o clone do repositório.
$ git clone https://github.com/natorjunior/computer_vision_response_exercises.git
#Entre no diretório '3 - Model Deployment (WebDocker)'/flask_code
$ cd computer_vision_response_exercises/3\ -\ Model\ Deployment\ \(WebDocker\)/flask_code
#Faça o build do Dockerfile
$ docker build -t web_aplication_flask:latest .
#Faça a execução do container 
$ docker run -d -p 5000:5000 web_aplication_flask:latest
```
###### Dockerfile:
```bash
FROM python:3.6
ADD . /code
WORKDIR /code
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]
```
#### Fazendo a requisição via Postman:
<img src="3%20-%20Model%20Deployment%20(WebDocker)/img/postman_params.PNG" width=320>
#### Fazendo a requisição via Browser:
<img src="3%20-%20Model%20Deployment%20(WebDocker)/img/result_browser.PNG" width=320>

Methods flask function
```python
#função para calcular o boundbox da imagem 
def process_image(path_img):
    return path_new_img
#função para renderizar o formulario de upload
@app.route('/upload')
def upload_file():
    return render_template('upload.html')
#função para processar e responder a imagem processada
@app.route('/process_file', methods = ['GET', 'POST'])
def respinse_upload():
    return send_file(path_img, mimetype='')
```

###### Para execução do notebook separadamente, instale as dependências :

```sh
$ pip install -U requeriments.txt
```
or 

```sh
$ pip install flask
$ pip install Werkzeug
$ pip install matplotlib
$ pip install opencv-contrib-python
```

Usage libs
```python
from flask import Flask, render_template, request,send_file
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from flask_code.ml.YOLO_V3 import YOLO_V3 #lib local
import numpy as np
import hashlib
import cv2
import os
```

## 4 - Object detection com deep learning
Foi implementado uma rede YOLOV3 para detecção de objetos, algumas imagens do conjunto de dados COCO de testes são usadas como demostração. A yolov3 tem seu potencial no reconhecimento de objetos devido a sua velocidade.

De acordo com informações extraidas do [paper](https://arxiv.org/abs/1804.02767) a YOLOV3 tem desempenho inferior a redes como a [RetinaNet](https://arxiv.org/abs/1708.02002), porém com velocidades superior. Podendo chegar a proximo de 57 fps como a versão adaptada do [paper](https://ieeexplore.ieee.org/abstract/document/8839032), chamada de mini-yolov3.
<img src="/4 - Object detection com deep learning/img/predict1.png" width=320>
<img src="/4 - Object detection com deep learning/img/predict2.png" width=320>
<img src="/4 - Object detection com deep learning/img/predict3.png" width=320>
<img src="/4 - Object detection com deep learning/img/predict4.png" width=320>

###### Install dependences:
```sh
$ pip install tensorflow
$ pip install keras
```


### limitações dos testes 
- Retreinamento da rede, com estatísticas de treinamento, teste e validação.
- Falta uma análise detalhada de desempenho em conjunto de dados desconhecidos.
- Seria interesante um benchmark gpu e cpu.


# 5) - Context segmentation com deep learning (Avançado)
Implementação parcial


