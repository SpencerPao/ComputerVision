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

import pandas as pd
import random
import time
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
    def __init__(self, action_size: int):
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
            300, 300, 1)))
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
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(learning_rate=self.learning_rate),
                      metrics=['accuracy'])
        print("We finish building the model")
        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min
        print(model.summary())
        return model

    def save_model(self):
        """
            Saves model weights to file location.
        """
        self.brain.save(self.weight_backup)

    def act(self, state: np.ndarray) -> int:
        """
            Parameters:
            ----------
            state: np.ndarray:
                Screenshot of gym.
            Return:
            ----------
            maximum value of which action to choose.
        """
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.brain.predict(state)
        # print("Predicted Act Values", act_values)
        return np.argmax(act_values[0])

    def remember(self, state: np.ndarray,
                 action: int,
                 reward: float,
                 next_state: np.ndarray,
                 done: bool) -> None:
        """
            Stores dinosaur state, action, reward, next_state, done into a deque.
            Uses stored values to retrain the model.
            Parameters:
            ----------
            state: np.ndarray
                screenshot of gym.
            action: int
                integer value 0 or 1 that determines whether dinosaur jumps or does nothing.
            reward: float
                current incremental reward assigned for each action of dinosaur.
            done: boolean
                Yes/or if dinosaur continues to run or if the game if over.

            Return:
            ----------
            Nothing.
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size) -> None:
        """
            Parameters:
            ----------
            sample_batch_size: int
                Size of observations to be used as refit input.
            Return:
            ----------
            None.
        """
        # print("Refitting Reinforcement learning model")
        # print("Memory Length", len(self.memory), "Sample Batch size: ", sample_batch_size)
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0])
            target_f = self.brain.predict(state)
            target_f[0][action] = target
            self.brain.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay


class TRexRunner:
    """
        TRexRunner is the agent that runs through the game.
    """

    def __init__(self):
        self.sample_batch_size = 32
        self.episodes = 10000
        self.env = gym.make("DinoRun-v0")
        self.action_size = self.env.action_space.n
        self.agent = Agent(self.action_size)

    def run(self):
        """ Step through the environment. """
        arr = []
        try:
            for index_episode in range(self.episodes):
                state = self.env.reset()
                done = False
                index = 0
                score = 0
                while not done:
                    action = self.agent.act(state)
                    next_state, reward, score, done = self.env.step(action)
                    print("Reward awarded:", reward, "Action taken: ", action)
                    self.agent.remember(state, action, reward, next_state, done)
                    if done:  # issue with scoring.
                        print(index)
                        for i in range(int(index/5)):  # for more "balance".
                            self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    index += 1
                print("Episode #{} Index Iteration: {} Ending Score: {} ".format(
                    index_episode, index + 1, score))
                arr.append(score)
                self.agent.replay(self.sample_batch_size)
            pd.DataFrame(arr).to_csv("results/scores.csv")
        finally:
            self.agent.save_model()


def main():
    """ Main executable for RL on dinosaur game. """
    dino = TRexRunner()
    dino.run()

    # keras.backend.set_image_data_format('channels_first')
    # env = gym.make("DinoRun-v0")
    # compute_train = True

    # episodes = 10
    # for episode in range(1, episodes+1):
    #     env.reset()
    #     done = False
    #     game_score = 0
    #     ending_score = 0
    #     c = 0
    #     while not done:
    #         action = env.action_space.sample()  # random action taken.
    #         screen, reward, score, done = env.step(action)
    #         ending_score += reward
    #         game_score = score
    #         im = Image.fromarray(screen)
    #         im.save("test_{}_{}.png".format(episode, c))
    #         c += 1
    #     print('Episode Number:{} Reward Accumulation:{} Final Game Score:{}'.format(
    #         episode, round(ending_score, 3), game_score))
    #     break

    # print("Closing Environment")
    # env.close()


if __name__ == '__main__':
    main()
