import glob

import gym
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D
from keras.optimizers import adam_v2

import tensorflow as tf
from rl.callbacks import ModelIntervalCheckpoint
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

env = gym.make('ALE/Breakout-v5', render_mode='human')
height, width, channels = env.observation_space.shape
actions = env.action_space.n  # number of Actions

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
  tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])

def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(3, height, width, channels)))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))  # strides is 1x1
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

model = build_model(height, width, channels, actions)

max_steps_per_episode = 10000
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
learning_rate = 1e-4

def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=epsilon_max, value_min=epsilon_min,
                                  value_test=.2,
                                  nb_steps=max_steps_per_episode)
    memory = SequentialMemory(limit=1000, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, enable_dueling_network=True, dueling_type='avg',
                   nb_actions=actions, nb_steps_warmup=1000)
    return dqn

dqn = build_agent(model, actions)
dqn.compile(adam_v2.Adam(learning_rate=learning_rate))
list_of_files = glob.glob('model_it_1/*')
#input("Press Enter to continue...")

done = 22 + 16
iteras = 108
for i in range(iteras - done):
    stepcount = (i+1) * 3000
    dqn.load_weights(f'model_it_1/dqn_model_weights_{stepcount}.h5')
    env.reset()
    scores = dqn.test(env, visualize=False, nb_max_episode_steps=300, verbose=0, nb_episodes=10)
    print(np.mean(scores.history['episode_reward']))
