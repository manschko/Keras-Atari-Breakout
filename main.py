"""
pip install tensorflow gym keras-rl2 gym[atari]

"""

import gym
import tensorflow as tf

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


"""
Radom choice




for episode in range(episodes):
    state = env.reset()
    done = False
    score = 0

    while not done:
        # env.render()  # prints the enviroment to the scree
        action = random.choice([0, 1, 2, 3])  # random action index
        n_sate, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()
"""
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
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


model = build_model(height, width, channels, actions)
# model.summary()#shows model architecture

"""
Build Agent
"""


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


"""
train and save Model
"""
def build_callbacks():
    checkpoint_weights_filepath = 'model/dqn_model_weights_{step}.h5'
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filepath, interval=1000)]
    return callbacks


dqn = build_agent(model, actions)
dqn.compile(adam_v2.Adam(learning_rate=learning_rate))

callbacks = build_callbacks()
dqn.fit(env, nb_steps=1000000, visualize=False, verbose=2, callbacks=callbacks)
