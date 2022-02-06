'''
    Creating the Model inspiration: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    https://medium.com/@gtnjuvin/my-journey-into-deep-q-learning-with-keras-and-gym-3e779cc12762
    https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    https://blog.paperspace.com/dino-run/
'''
import pickle
import gym_dinorun
import gym
import numpy as np
import os

import pandas as pd
import random
from collections import deque

import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation
from tensorflow.keras.optimizers import Adam


class Agent():
    def __init__(self, action_size: int):
        self.weight_backup = "models/dino_runner.h5"
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.learning_rate = 0.001
        self.OBSERVATION = 100  # timesteps to observe before training
        self.EXPLORE = 100000  # frames over which to anneal epsilon
        self.FINAL_EPSILON = 0.0001  # final value of epsilon
        self.INITIAL_EPSILON = 0.1  # starting value of epsilon
        self.REPLAY_MEMORY = 50000  # number of previous transitions to remember
        self.FRAME_PER_ACTION = 1
        self.gamma = 0.95
        # self.exploration_rate = 1.0
        self.exploration_min = 0.01
        self.exploration_decay = 0.995
        self.img_rows, self.img_cols = 300, 300
        self.img_channels = 4  # Number of stacked images
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
        model.add(Conv2D(32, (8, 8), padding='same', strides=(4, 4),
                  input_shape=(self.img_rows, self.img_cols, self.img_channels)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2),  padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1),  padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(self.action_size))
        # Where Q learning occurs. (ACTIONS) Number of actions dino can do.
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        print("We finish building the model")
        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min
        # print(model.summary())
        return model

    def save_model(self):
        """
            Saves model weights to file location.
        """
        self.brain.save(self.weight_backup)

    def act(self, state: np.ndarray, epsilon: float) -> int:
        """
            Parameters:
            ----------
            state: np.ndarray:
                4 x Screenshots of gym.

            epsilon: float
                threshold to determine whether random use is warrented.

            Return:
            ----------
            maximum value of which action to choose.
        """
        if np.random.rand() <= epsilon:
            print("----------Random Action----------", "epsilon: ", epsilon)
            return random.randrange(self.action_size)

        act_values = self.brain.predict(state)
        # print("Predicted Act Values", act_values)
        return np.argmax(act_values)  # return action index.

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

    def replay(self, model, minibatch, inputs, targets):
        """
            Parameters:
            ----------
            model:
                DQN model.
            minibatch:
                Random images to use for the training.
            inputs:
                Value to use as X variable.
            targets:
                Value to use as Y predictor.
            done: bool:
                determine whether the game has ended.

            Return:
            ----------
            loss, q_result.
        """
        for i in range(0, len(minibatch)):
            loss = 0
            state_t = minibatch[i][0]    # 4D stack of images
            action_t = minibatch[i][1]  # This is action index
            reward_t = minibatch[i][2]  # reward at state_t due to action_t
            state_t1 = minibatch[i][3]  # next state
            # wheather the agent died or survided due the action
            done = minibatch[i][4]
            inputs[i:i + 1] = state_t
            targets[i] = model.predict(state_t)  # predicted q values
            q_result = model.predict(state_t1)  # predict q values for next step
            if done:
                targets[i, action_t] = reward_t  # if terminated, only equals reward
            else:
                targets[i, action_t] = reward_t + self.gamma * np.max(q_result)
        loss += model.train_on_batch(inputs, targets)
        return loss, q_result


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
        self.init_cache()

    def save_obj(self, obj, name):
        with open('objects/' + name + '.pkl', 'wb') as f:  # dump files into objects folder
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, name):
        with open('objects/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def init_cache(self):
        """initial variable caching, done only once"""
        self.save_obj(self.agent.INITIAL_EPSILON, "epsilon")
        t = 0
        self.save_obj(t, "time")
        D = deque()
        self.save_obj(D, "D")

    def save_status(self, model,
                    D: deque,
                    t: int,
                    epsilon: float,
                    loss_df: pd.DataFrame,
                    scores_df: pd.DataFrame,
                    actions_df: pd.DataFrame,
                    q_values_df: pd.DataFrame):
        model.save_weights(self.agent.weight_backup, overwrite=True)
        self.save_obj(D, "D")  # saving episodes
        self.save_obj(t, "time")  # caching time steps
        # cache epsilon to avoid repeated randomness in actions
        self.save_obj(epsilon, "epsilon")
        loss_df.to_csv("./objects/loss_df.csv", index=False)
        scores_df.to_csv("./objects/scores_df.csv", index=False)
        actions_df.to_csv("./objects/actions_df.csv", index=False)
        q_values_df.to_csv("./objects/q_values.csv", index=False)

    def run(self, loss_df, scores_df, actions_df, q_values_df, compute_train: bool):
        """ Step through the environment.

            Parameters:
            ----------
            loss_df: pd.DataFrame
                Dataframe for loss values
            scores_df: pd.Dataframe
                DataFrame for scores.
            actions_df: pd.Dataframe
                DataFrame for actions.
            q_values_df: pd.Dataframe
                DataFrame for q values.
            compute_train: bool
                True: Train Model, False: Use Model out of box.

            Return:
            ----------
            None.
        """
        # last_time = time.time()
        # load from file system; # store the previous observations in replay memory
        D = self.load_obj("D")
        try:
            # D = deque()  # experience replay memory
            state = self.env.reset()  # initial state.
            do_nothing = np.zeros(self.action_size)
            do_nothing[0] = 0  # 0 -> None, 1 -> Jump
            screen, reward, score, done = self.env.step(int(do_nothing[0]))
            # Stack 4 images to create placeholder input
            stacked_images = np.stack((screen, screen, screen, screen), axis=2)
            # Reshape as observation
            stacked_images = stacked_images.reshape(1,  # observation
                                                    stacked_images.shape[0],  # height
                                                    stacked_images.shape[1],  # width
                                                    stacked_images.shape[2])  # number images
            # print("Stacked images shape: ", stacked_images.shape)
            initial_state = stacked_images
            if compute_train:  # Train
                OBSERVE = self.agent.OBSERVATION
                epsilon = self.load_obj("epsilon")
                model = self.agent.brain
                model.summary()
                print("Regular Model has been loaded in successfully.")
            else:
                OBSERVE = 999999999
                epsilon = self.agent.FINAL_EPSILON
                model = self.agent.brain
                print("Trained Model has been loaded in successfully.")
            t = self.load_obj("time")  # time step.

            # for index_episode in range(self.episodes):
            while (True):
                loss, q_result, action_index, reward_t = 0, 0, 0, 0
                action_t = np.zeros([self.action_size])  # action at t
                q = model.predict(stacked_images)  # input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)         # chosing index with maximum q value
                action_index = max_Q
                action_t[action_index] = 1        # o=> do nothing, 1=> jump
                max_action = max(action_t)
                max_index = np.where(action_t == max_action)
                actions_df.loc[len(actions_df)] = max_index[0][0]  # determining the action taken.
                # run the selected action and observed next state and reward
                x_t1, reward_t, score, terminal = self.env.step(max_index[0][0])
                scores_df.loc[len(scores_df)] = score
                x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # 1x20x40x1
                # append the new image to input stack and remove the first one
                stacked_images1 = np.append(x_t1, stacked_images[:, :, :, :3], axis=3)

                # store the transition
                D.append((stacked_images, action_index, reward_t, stacked_images1, terminal))
                if t > OBSERVE and terminal:
                    minibatch = random.sample(D, self.sample_batch_size)
                    inputs = np.zeros((self.sample_batch_size,
                                       stacked_images.shape[1],
                                       stacked_images.shape[2],
                                       stacked_images.shape[3]))
                    targets = np.zeros((inputs.shape[0], self.action_size))  # 32, 2
                    loss, q_result = self.agent.replay(model, minibatch, inputs, targets)
                    print("Loss: ", loss, "Q Result: ", q_result)
                    print("np.max: ", np.max(q_result))
                    print("Length Q values Df", len(q_values_df))
                    loss_df.loc[len(loss_df)] = loss
                    # print("Value to include : ", q_values_df.loc[len(q_values_df)-1])
                    q_values_df.loc[len(q_values_df)] = np.argmax(q_result)
                    state = ""
                    if t <= OBSERVE:
                        state = "observe"
                    elif t > OBSERVE and t <= OBSERVE + self.agent.EXPLORE:
                        state = "explore"
                    else:
                        state = "train"
                    print("Trained on End Screen:",
                          "TIMESTEP", t,
                          "/ STATE", state,
                          "/ EPSILON", epsilon,
                          "/ ACTION", action_index,
                          "/ REWARD", reward_t,
                          "/ Q_MAX ", np.argmax(q_result),
                          "/ Loss ",  loss)
                    self.env.reset()
                stacked_images = stacked_images1
                t += 1

            # # if done:
            # #     self.env.resume()
            # loss, q_result, action_index, reward_t, score = 0, 0, 0, 0, 0
            # action_range = np.zeros([self.action_size])  # action at t (None, jump)
            # done = False
            # # choose an action epsilon greedy
            # if t % self.agent.FRAME_PER_ACTION == 0:  # parameter to skip frames for actions
            #     action_index = self.agent.act(stacked_images, epsilon)
            #     action_range[action_index] = 1
            #
            # if epsilon > self.agent.FINAL_EPSILON and t > OBSERVE:
            #     epsilon -= (self.agent.INITIAL_EPSILON -
            #                 self.agent.FINAL_EPSILON) / self.agent.EXPLORE
            #
            # # run the selected action and observed next state and reward_t
            # max_action = max(action_range)
            # max_index = np.where(action_range == max_action)
            # next_state, reward_t, score, done = self.env.step(max_index[0][0])
            # # helpful for measuring frame rate
            # # print('fps: {0}'.format(1 / (time.time()-last_time)))
            # # last_time = time.time()
            # next_state = next_state.reshape(
            #     1, next_state.shape[0], next_state.shape[1], 1)  # 1x300x300x1
            # # append the new image to input stack and remove the first one
            # stacked_images1 = np.append(next_state, stacked_images[:, :, :, :3], axis=3)
            #
            # # store the transition in D
            # D.append((stacked_images, action_index, reward_t, stacked_images1, done))
            # if len(D) > self.agent.REPLAY_MEMORY:
            #     D.popleft()
            # # only train if done observing
            # if t > OBSERVE:
            #     # sample a minibatch to train on
            #     minibatch = random.sample(D, self.sample_batch_size)
            #     inputs = np.zeros(
            #         (self.sample_batch_size,
            #          stacked_images.shape[1],
            #          stacked_images.shape[2],
            #          stacked_images.shape[3]))
            #     targets = np.zeros((inputs.shape[0], self.agent.action_size))  # 32, 2
            #     loss = self.agent.replay(model, minibatch, inputs, targets)
            #     loss_df.loc[len(loss_df)] = loss
            #     q_values_df.loc[len(q_values_df)] = np.max(q_result)
            #
            # t = t + 1
            # stacked_images = stacked_images1
            # # save progress every 1000 iterations
            # if t % 1000 == 0:
            #     print("-------------- Saving state --------------")
            #     self.env.pause()  # pause game while saving to filesystem
            #     self.save_status(model, D, t, epsilon, loss_df,
            #                      scores_df, actions_df, q_values_df)
            #     self.env.resume()
            # state = ""
            # if t <= OBSERVE:
            #     state = "observe"
            # elif t > OBSERVE and t <= OBSERVE + self.agent.EXPLORE:
            #     state = "explore"
            # else:
            #     state = "train"
            # print("TIMESTEP", t,
            #       "/ STATE", state,
            #       "/ EPSILON", epsilon,
            #       "/ ACTION", action_index,
            #       "/ REWARD", reward_t,
            #       "/ Q_MAX ", np.max(q_result),
            #       "/ Loss ",  loss)

            print("Episode finished!")
            print("************************")
        finally:
            print('******************************************************')
            print("Manual Override.... saving state.")
            self.save_status(model, D, t, epsilon, loss_df, scores_df, actions_df, q_values_df)
            print('- Finished Saving.')
            print('******************************************************')


def main():
    """ Main executable for RL on dinosaur game. """
    # path variables
    loss_file_path = "./objects/loss_df.csv"
    actions_file_path = "./objects/actions_df.csv"
    q_value_file_path = "./objects/q_values.csv"
    scores_file_path = "./objects/scores_df.csv"
    # Intialize log structures from file if exists else create new
    loss_df = pd.read_csv(loss_file_path) if os.path.isfile(
        loss_file_path) else pd.DataFrame(columns=['loss'])
    scores_df = pd.read_csv(scores_file_path) if os.path.isfile(
        loss_file_path) else pd.DataFrame(columns=['scores'])
    actions_df = pd.read_csv(actions_file_path) if os.path.isfile(
        actions_file_path) else pd.DataFrame(columns=['actions'])
    q_values_df = pd.read_csv(actions_file_path) if os.path.isfile(
        q_value_file_path) else pd.DataFrame(columns=['qvalues'])

    dino = TRexRunner()
    dino.run(loss_df,
             scores_df,
             actions_df,
             q_values_df,
             compute_train=True)


if __name__ == '__main__':
    main()
