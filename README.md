# Keras-Atari-Breakout-
Creating a DQN network under basis for the deepmind paper https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

Implementation with TensorFlow 2 and Keras-RL2

main.py includes everything for running witch python 3.10+

main_docker.py is the implementation with Python 3.6 for the docker container running on a Jetson device

# Setup
install python package

```pip install -r req.txt```

### install ROMs
download Roms from http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html

urar them

atari-py (python version 3.7-)

``python -m atari_py.import_roms <path to Rom folder>``

ALE-py (python version 3.8+)

``ale-import-roms <path to Rom folder>``

# Start
python version: 3.8+

```python main.py```

python version: 3.7-

```python main_docker.py```

visualizer python 3.10+

```python Visualize_Model.py```

# docker
build

```docker build -t <name> .```

run

```dcoker run --gpus all <name>```