# base container for jetsonNano Tensorflow
FROM tensorflow/tensorflow:latest-gpu-jupyter
#RUN apt-get update -qq \
#    && apt-get upgrade -qq




COPY . /src
WORKDIR /src
#getting ale-py to run in docker
#RUN pip3 install --upgrade pip
RUN pip install --trusted-host pypi.python.org -r requirements.txt
#import ROMS
RUN ale-import-roms ROMS2/
# CMD ['python3', '/src/main.py']
EXPOSE 8080