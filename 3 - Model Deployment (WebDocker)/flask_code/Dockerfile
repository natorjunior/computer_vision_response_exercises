FROM python:3.6
ADD . /code
WORKDIR /code
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install nano
RUN wget -O ml/yolov3.weights https://pjreddie.com/media/files/yolov3.weights
ENTRYPOINT ["python"]
CMD ["app.py"]
