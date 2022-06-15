from keras.layers import Dense, Flatten, Convolution2D, Conv2D
from keras.models import Sequential
from rl.agents import DQNAgent
from rl.callbacks import Callback
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

max_steps_per_episode = 10000
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter


def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(3, height, width, channels)))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))  # strides is 1x1
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    model.summary()
    return model


def build_model_keras(height, width, actions):
    # Convolutions on the frames on the screen
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(3, height, width, 1)))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(actions))
    model.summary()

    return model


def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=epsilon_max, value_min=epsilon_min,
                                  value_test=.2,
                                  nb_steps=max_steps_per_episode)
    memory = SequentialMemory(limit=1000, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, enable_dueling_network=True, dueling_type='avg',
                   nb_actions=actions, nb_steps_warmup=1000)
    return dqn


class Callback(Callback):
    def __init__(self, filepath, interval):
        super().__init__()
        self.filepath = filepath + 'a'
        self.interval = interval
        self.total_episodes = 2500

    def on_episode_end(self, episode, logs={}):
        """ Save weights at interval steps during training """
        self.total_episodes += 1
        if self.total_episodes % self.interval != 0:
            # Nothing to do.
            return

        filepath = self.filepath.format(episode=self.total_episodes, **logs)
        self.model.save_weights(filepath, overwrite=True)
