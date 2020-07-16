FROM python:3.6
ADD . /code
WORKDIR /code
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install nano
ENTRYPOINT ["python"]
CMD ["app.py"]
