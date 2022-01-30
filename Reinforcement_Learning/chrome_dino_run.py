'''
    Creating the Model inspiration: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    https://medium.com/@gtnjuvin/my-journey-into-deep-q-learning-with-keras-and-gym-3e779cc12762
    https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

'''
import gym
import gym_dinorun
import math
import numpy as np
import os

import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import keras
import tensorflow as tf
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from PIL import Image

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


class Agent():
    def __init__(self, action_size):
        self.weight_backup = "models/dino_runner.h5"
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.exploration_rate = 1.0
        self.exploration_min = 0.01
        self.exploration_decay = 0.995
        self.brain = self._buildmodel()

    def _buildmodel(self):
        """
            Constructs tensorflow model.

            Return:
            ----------
            Tensorflow model
        """
        print("Now we build the model")
        model = Sequential()
        model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(
            64, 64, 4)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        # Where Q learning occurs. (ACTIONS) Number of actions dino can do.
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        print("We finish building the model")
        # if os.path.isfile(self.weight_backup):
        #     model.load_weights(self.weight_backup)
        #     self.exploration_rate = self.exploration_min
        print(model.summary())
        return model

    def save_model(self):
        self.brain.save(self.weight_backup)

    def act(self, state):
        """
            Parameters:
            ----------
            state: np.ndarray:
                Screenshot of gym.
            Return:
            ----------
            maximum value.
        """
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.brain.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size):
        # print("Refitting Reinforcement learning model")
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
                #print("REPLAY State", next_state.shape)
                # print(self.brain.predict(next_state)[0])
                target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0])
            target_f = self.brain.predict(state)
            target_f[0][action] = target
            self.brain.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay


class TRexRunner:
    def __init__(self):
        self.sample_batch_size = 32
        self.episodes = 10000
        self.env = gym.make("DinoRun-v0")
        self.action_size = self.env.action_space.n
        self.agent = Agent(self.action_size)

    def run(self):
        try:
            for index_episode in range(self.episodes):
                state = self.env.reset()
                done = False
                index = 0
                score = 0
                while not done:
                    action = self.agent.act(state)
                    next_state, reward, score, done = self.env.step(action)
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    index += 1
                print("Episode #{} Score: {} Current Score {}".format(index_episode, index + 1, score))
                self.agent.replay(self.sample_batch_size)
        finally:
            self.agent.save_model()


def main():
    # keras.backend.set_image_data_format('channels_first')
    dino = TRexRunner()
    dino.run()
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # env = gym.make("DinoRun-v0")
    # compute_train = True

    # Not working...
    # if compute_train:
    #     model_path_to_save = 'models/'
    #     model = buildmodel()
    #     model.summary()
    #     dqn = build_agent(model, actions=3)
    #     dqn.compile(optimizer=Adam(learning_rate=1e-4), metrics=['mae'])
    #     print("________________Here__________________________________________")
    #     dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

    # episodes = 10
    # for episode in range(1, episodes+1):
    #     env.reset()
    #     done = False
    #     game_score = 0
    #     ending_score = 0
    #     # c = 0
    #     while not done:
    #         action = env.action_space.sample()  # random action taken.
    #         screen, reward, score, done = env.step(action)
    #         ending_score += reward
    #         game_score = score
    #         print(screen.shape)
    #         # im = Image.fromarray(screen)
    #         # im.save("test_{}_{}.jpeg".format(episode, c))
    #         # c += 1
    #     print('Episode Number:{} Reward Accumulation:{} Final Game Score:{}'.format(
    #         episode, round(ending_score, 3), game_score))
    #     break

    # for _ in range(1000):  # Number of "steps": Make steps large for training.
    #     # env.render()
    #     screen, reward, score, done = env.step(env.action_space.sample())  # take a random action
    #     if _ == 0:
    #         init_screen = screen
    #         screen_height, screen_width, color = init_screen.shape
    #         print(screen_height, screen_width, color)
    #     else:
    #         current_screen = screen
    #         state = current_screen - last_screen
    #
    #     print(screen.shape)
    #     print('Step: ', _, 'Reward: ', reward, 'Score:', score, 'Done:', done)
    #     break

    # print("Closing Environment")
    # env.close()


if __name__ == '__main__':
    main()
