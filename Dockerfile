# base container for jetsonNano Tensorflow
FROM l4t-tensorflow:r34.1.0-tf1.15-py3
RUN apt-get update -qq \
    && apt-get upgrade -qq \
    && apt-get install unrar




COPY . /src
WORKDIR /src
#getting ale-py to run in docker
#RUN pip3 install --upgrade pip
RUN pip install --trusted-host pypi.python.org -r requirements.txt
#import ROMS
RUN wget http://www.atarimania.com/roms/Roms.rar
RUN unrar Roms.rar
RUN python3 -m atari_py.import_roms Roms/
RUN git clone https://github.com/openai/baselines.git
RUN pip install -e baselines/
# CMD ['python3', '/src/main.py']
EXPOSE 8080