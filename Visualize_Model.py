import glob

import gym
import numpy as np
from gym.wrappers import FrameStack
from keras.optimizers import adam_v2
from stable_baselines3.common.atari_wrappers import AtariWrapper

import util

env = gym.make('ALE/Breakout-v5', render_mode='human')

actions = env.action_space.n  # number of Actions
print(env.observation_space.shape)
env2 = AtariWrapper(env)
print(env2.observation_space.shape)
env3 = FrameStack(env2, 4)
print(env3.observation_space.shape)
frames, height, width, channels = env3.observation_space.shape

env3 = env2
model = util.build_model_keras(height, width, actions)

max_steps_per_episode = 10000
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
learning_rate = 1e-4

dqn = util.build_agent(model, 4)
dqn.compile(adam_v2.Adam(learning_rate=learning_rate))

"""
get weights to Visualize
"""
choice = input("Do you want to Visualize multible weights (yes) or just one (no) \n")

files = []
skip_amount = 1

# python 3.10 req
match choice:
    case 'yes':
        folder_path = input("Folder path to Weight files:\n")
        files = glob.glob(folder_path + "/*")
        skip_amount_text = input("how many file do you want to skip on each test? default = 1\n")
        try:
            skip_amount = int(skip_amount_text)
            if skip_amount <= 1:
                skip_amount = 1
        except:
            skip_amount = 1

    case 'no':
        files.append(input('Path to wieght file\n'))

amount_of_episodes = 1
input = input('how many episode do you want to test on each file\n')
try:
    amount_of_episodes = int(input)
    if amount_of_episodes <= 1:
        amount_of_episodes = 1
except:
    amount_of_episodes = 1

for i in range(int(len(files) / skip_amount)):
    print('test')
    dqn.load_weights(files[i * skip_amount])
    env.reset()
    scores = dqn.test(env3, visualize=False, nb_max_episode_steps=300, verbose=0, nb_episodes=amount_of_episodes)
    print(np.mean(scores.history['episode_reward']))
