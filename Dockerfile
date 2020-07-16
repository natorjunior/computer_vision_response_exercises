FROM python:3.6
ADD . /jupyter_notebook
WORKDIR /jupyter_notebook
RUN pip install -r requirements.txt
RUN pip install notebook
RUN apt-get update
RUN apt-get install nano
RUN python3 down_data.py
ENTRYPOINT ["jupyter-notebook"]
CMD ["--no-browser"]
