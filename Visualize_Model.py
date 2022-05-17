import glob

import gym
import numpy as np
from keras.optimizers import adam_v2

import main

env = gym.make('ALE/Breakout-v5', render_mode='human')
height, width, channels = env.observation_space.shape
actions = env.action_space.n  # number of Actions

model = main.build_model(height, width, channels, actions)

max_steps_per_episode = 10000
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
learning_rate = 1e-4

dqn = main.build_agent(model, actions)
dqn.compile(adam_v2.Adam(learning_rate=learning_rate))

"""
get weights to Visualize
"""
choice = input("Do you want to Visualize multible weights (yes) or just one (no)")

files = []
skip_amount = 1

# python 3.10 req
match choice:
    case 'yes':
        files = glob.glob(folder_path)
        folder_path = input("Folder path to Weight files:")
        skip_amount_text = input("how many file do you want to skip on each test? default = 1")
        try:
            skip_amount = int(skip_amount_text)
            if skip_amount <= 1:
                skip_amount = 1
        except:
            skip_amount = 1

    case 'no':
        files.append(files, input('Path to wieght file'))

amount_of_episodes = 1
input = input('how many episode do you want to test on each file')
try:
    amount_of_episodes = int(input)
    if amount_of_episodes <= 1:
        amount_of_episodes = 1
except:
    amount_of_episodes = 1

for i in range(int(len(files) / skip_amount)):
    dqn.load_weights(files[i * skip_amount])
    env.reset()
    scores = dqn.test(env, visualize=False, nb_max_episode_steps=300, verbose=0, nb_episodes=amount_of_episodes)
    print(np.mean(scores.history['episode_reward']))


