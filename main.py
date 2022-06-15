"""
pip install tensorflow gym keras-rl2 gym[atari]
"""
import glob
import os

import gym
from gym.wrappers import GrayScaleObservation, ResizeObservation
from keras.optimizers import adam_v2

import util
from util import Callback

env = gym.make('ALE/Breakout-v5', render_mode=None)
env2 = GrayScaleObservation(env)
env2 = ResizeObservation(env2, (84, 84))
env = env2

try:
    height, width = env.observation_space.shape
except:
    height, width, channels = env.observation_space.shape
actions = env.action_space.n  # number of Actions

episodes = 5
file_path = input("Path to Weight file if continuing training, or enter if no file\n")

model = util.build_model_keras(height, width, actions)


count = 0
latest_file = ''
try:
    list_of_files = glob.glob('model/*')  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    count = int(latest_file.split('_')[-1].split('.')[0])
except:
    count = 0

steps = 6000000


def build_callbacks():
    checkpoint_weights_filepath = 'model/dqn_model_weights_{episode}.h5'
    callbacks = [Callback(checkpoint_weights_filepath, interval=100)]
    return callbacks


learning_rate = 0.00025
dqn = util.build_agent(model, actions)
dqn.compile(adam_v2.Adam(learning_rate=learning_rate))

if file_path != '':
    dqn.load_weights(file_path)

callbacks = build_callbacks()
dqn.fit(env, nb_steps=steps, visualize=False, verbose=2, callbacks=callbacks)
