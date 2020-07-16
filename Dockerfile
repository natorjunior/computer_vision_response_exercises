ARG BASE_CONTAINER=jupyter/scipy-notebook
FROM $BASE_CONTAINER
ADD . /jupyter_notebook
WORKDIR /jupyter_notebook
RUN pip install flask
RUN pip install Werkzeug
RUN pip install matplotlib
RUN pip install opencv-contrib-python
RUN pip install tensorflow
RUN python down_data.py