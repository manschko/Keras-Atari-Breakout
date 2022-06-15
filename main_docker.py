import gym
from gym.wrappers import GrayScaleObservation, ResizeObservation

import util
from util import Callback

env = gym.make('BreakoutNoFrameskip-v4')
env2 = GrayScaleObservation(env)
env2 = ResizeObservation(env2, (84, 84))
env = env2
height, width = env.observation_space.shape
actions = env.action_space.n  # number of Actions
episodes = 5

model = util.build_model_keras(height, width, actions)


def build_callbacks():
    checkpoint_weights_filepath = 'model/dqn_model_weights_{step}.h5'
    callbacks = [Callback(checkpoint_weights_filepath, interval=1000)]
    return callbacks


callbacks = build_callbacks()
steps = 6000000
learning_rate = 0.00025
dqn = util.build_agent(model, actions)
dqn.fit(env, nb_steps=steps, visualize=False, verbose=2, callbacks=callbacks)
