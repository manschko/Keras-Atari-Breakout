"""
pip install tensorflow gym keras-rl2 gym[atari]

"""
import glob
import os

import gym
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D
from keras.optimizers import adam_v2

from rl.callbacks import ModelIntervalCheckpoint
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy


env = gym.make('ALE/Breakout-v5', render_mode=None)
height, width, channels = env.observation_space.shape
actions = env.action_space.n  # number of Actions
episodes = 5
file_path = input("Path to Weight file if continuing training, or enter if no file")

"""
create DeepQ Model
"""


def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(3, height, width, channels)))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))  # strides is 1x1
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


model = build_model(height, width, channels, actions)
model.summary()#shows model architecture

"""
Build Agent
"""


max_steps_per_episode = 10000
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
learning_rate = 0.00025


def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=epsilon_max, value_min=epsilon_min,
                                  value_test=.2,
                                  nb_steps=max_steps_per_episode)
    memory = SequentialMemory(limit=1000, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, enable_dueling_network=True, dueling_type='avg',
                   nb_actions=actions, nb_steps_warmup=1000)
    return dqn


"""
train and save Model
"""
count = 0
latest_file = ''
try:
    list_of_files = glob.glob('model/*') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    count = int(latest_file.split('_')[-1].split('.')[0])
except:
    count = 0

steps = 900000

def build_callbacks():

    checkpoint_weights_filepath = 'model/dqn_model_weights_{step}.h5'
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filepath, interval=1000)]
    return callbacks


dqn = build_agent(model, actions)
dqn.compile(adam_v2.Adam(learning_rate=learning_rate))

if file_path != '':
    dqn.load_weights(file_path)


callbacks = build_callbacks()
dqn.fit(env, nb_steps=steps, visualize=False, verbose=2, callbacks=callbacks)
