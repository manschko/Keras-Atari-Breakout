# base container for jetsonNano Tensorflow
FROM nvcr.io/nvidia/l4t-tensorflow:r32.7.1-tf2.7-py3
RUN apt-get update -qq \
    && apt-get upgrade -qq \
    && apt-get install --no-install-recommends -y \
    unrar \
    cmake \
    git

COPY . /src
WORKDIR /src
#install packages
RUN pip3 install --upgrade pip
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt
#check if 
#import ROMS
RUN wget http://www.atarimania.com/roms/Roms.rar
RUN locale-gen en_US.UTF-8
RUN unrar x Roms.rar -y
RUN AutoROM --install-dir /usr/local/lib/python3.6/dist-packages/atari_py/atari_roms -y
#import Baselines / AI enviroments
RUN mkdir model
# start Program
# CMD ['python3', '/src/main_docker.py']