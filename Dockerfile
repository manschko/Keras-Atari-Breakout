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
#getting ale-py to run in docker
RUN pip3 install --upgrade pip
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt
#import ROMS
RUN wget http://www.atarimania.com/roms/Roms.rar
RUN locale-gen en_US.UTF-8
RUN unrar x Roms.rar -y
#RUN python3 -m atari_py.import_roms ROMS/
RUN git clone https://github.com/openai/baselines.git
RUN pip3 install -e baselines/
# CMD ['python3', '/src/main.py']
EXPOSE 8080