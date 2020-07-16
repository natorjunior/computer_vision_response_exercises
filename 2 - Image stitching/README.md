# Solutions for computer vision exercises
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



