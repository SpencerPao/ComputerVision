"""
Creating the Model inspiration: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
https://medium.com/@gtnjuvin/my-journey-into-deep-q-learning-with-keras-and-gym-3e779cc12762
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
https://blog.paperspace.com/dino-run/
"""
import pickle
import gym_dinorun
import gym
import numpy as np
import os

from typing import Deque
import pandas as pd
import random
from collections import deque

#import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation
from tensorflow.keras.optimizers import Adam


class Agent():
    def __init__(self, action_size: int):
        self.weight_backup = "models/dino_runner.h5"
        self.action_size = action_size
        self.memory = deque(maxlen=5000)  # number of images.
        self.epsilon = 1  # starting value of epsilon
        self.epsilon_min = 0.1
        self.gamma = 0.95
        # self.exploration_rate = 1.0
        self.epsilon_decay = 0.995
        self.img_rows, self.img_cols, self.img_channels = 300, 300, 1
        self.update_rate = 1000  # number of iterations to update model.
        # Construct DQN models
        self.model = self._buildmodel()
        self.target_model = self._buildmodel()
        self.target_model.set_weights(self.model.get_weights())
        self.model.summary()

    def _buildmodel(self):
        """
        Constructs keras model.
            Uses optimizer Adam(learning_rate=0.01)
                loss:
                    MSE
            DQN Model:
                3 Convolution layers.
                FC: 512 nodes
                    2 output nodes
        Return:
        ----------
        Keras model
        """
        print("Now we build the model")
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=4, padding='same', input_shape=(
            self.img_rows, self.img_cols, self.img_channels)))
        model.add(Activation('relu'))

        model.add(Conv2D(64, (4, 4), strides=2, padding='same'))
        model.add(Activation('relu'))

        model.add(Conv2D(64, (3, 3), strides=1, padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())

        # FC Layers
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam())  # learning rate = 0.01
        print("We finish building the model")
        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
        return model

    def update_target_model(self) -> None:
        """Update the target model's weights with the prediction network weights."""
        self.target_model.set_weights(self.model.get_weights())

    def save_model(self) -> None:
        """Saves model weights to file location."""
        self.model.save(self.weight_backup)

    def act(self, state: np.ndarray) -> int:
        """
        Parameters:
        ------
        state: np.ndarray:
            A screen shot of the gym.

        Return:
        ------
        maximum value (index) of which action to choose.

        """
        if np.random.rand() <= self.epsilon:
            # print("----------Random Action----------", "epsilon: ", epsilon)
            return random.randrange(self.action_size)

        act_values = self.model.predict(state)
        # print("Predicted Act Values", act_values)
        return np.argmax(act_values[0])  # return action index.

    def remember(self, state: np.ndarray,
                 action: int,
                 reward: float,
                 next_state: np.ndarray,
                 done: bool) -> None:
        """
        Stores dinosaur state, action, reward, next_state, done into a deque.
        Uses stored values to fit the model.

        Parameters
        ----------
        state: np.ndarray
            screenshot of gym.
        action: int
            integer value 0 or 1 that determines whether dinosaur jumps or does nothing.
        reward: float
            current incremental reward assigned for each action of dinosaur.
        next_state: np.ndarray
            screenshot of gym (next image)
        done: boolean
            Yes/or if dinosaur continues to run or if the game if over.

        Return:
        ------
        None.

        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, minibatch: Deque):
        """
        Parameters:
        ----------
        minibatch: Deque
            Consists of features within the minibatch score_array
                state, action, reward, next_state, done
        Return:
        ----------
        loss, q_result.
        """
        for state, action, reward, next_state, done in minibatch:

            if not done:
                target = (reward + self.gamma * np.argmax(self.target_model.predict(next_state)))
            else:
                target = reward

            # Construct the target vector as follows:
            # 1. Use the current model to output the Q-value predictions
            target_f = self.model.predict(state)

            # 2. Rewrite the chosen action value with the computed target
            target_f[0][action] = target

            # 3. Use vectors in the objective computation
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


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
        self.save_obj(self.agent.epsilon, "epsilon")
        t = 0
        self.save_obj(t, "time")
        D = deque()
        self.save_obj(D, "D")

    def save_status(self, model=None):
        # D: deque = None,
        # t: int = None,
        # epsilon: float = None,
        # loss_df: pd.DataFrame = None,
        # scores_df: pd.DataFrame = None,
        # actions_df: pd.DataFrame = None,
        # q_values_df: pd.DataFrame = None
        # ):
        """
        model: keras model
            Model weights to be saved.
        """
        model.save_weights(self.agent.weight_backup, overwrite=True)
        # self.save_obj(D, "D")  # saving episodes
        # self.save_obj(t, "time")  # caching time steps
        # # cache epsilon to avoid repeated randomness in actions
        # self.save_obj(epsilon, "epsilon")
        # loss_df.to_csv("./objects/loss_df.csv", index=False)
        # scores_df.to_csv("./objects/scores_df.csv", index=False)
        # actions_df.to_csv("./objects/actions_df.csv", index=False)
        # q_values_df.to_csv("./objects/q_values.csv", index=False)

    def blend_images(self, images: Deque, blend: int):
        """
        Average the images in deque to produce stepped predictive state.
        Parameters
        ----------
        images: Deque
            Deque of hosted images.
        blend: int
            Number of images to take the average
        Return:
        ------
        averaged images
        """
        avg_image = np.expand_dims(np.zeros((300, 300, 1), np.float64), axis=0)  # observation first

        for image in images:
            avg_image += image

        if len(images) < blend:
            return avg_image / len(images)
        else:
            return avg_image / blend

    def run(self, loss_df, scores_df, actions_df, q_values_df, compute_train: bool):
        """
        Step through the environment.

        Parameters
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
        ------
        None.

        """
        total_time = 0
        blend = 4
        all_rewards = 0  # Used to compute avg reward over time
        try:
            for e in range(self.episodes):
                total_reward = 0
                game_score = 0
                state = self.env.reset()  # initial state.
                images = deque(maxlen=blend)
                images.append(state)
                for t in range(20000):
                    total_time += 1
                    # Every update_rate timesteps we update the target network parameters
                    if total_time % self.agent.update_rate == 0:
                        self.agent.update_target_model()
                    # Returning the average of the last 4 frames.
                    state = self.blend_images(images, blend)
                    action = self.agent.act(state)
                    next_state, reward_t, score, done = self.env.step(action)
                    # process next state.
                    images.append(next_state)
                    next_state = self.blend_images(images, blend)

                    # Store sequence in replay memory
                    self.agent.remember(state, action, reward_t, next_state, done)

                    state = next_state
                    game_score += reward_t
                    reward_t -= 1  # Punish behavior which does not accumulate reward
                    total_reward += reward_t
                    if done:
                        all_rewards += game_score

                        print("episode: {}/{}, game score: {}, reward: {}, avg reward: {}, time: {}, total time: {}"
                              .format(e+1, self.episodes, game_score, total_reward, all_rewards/(e+1), t, total_time))
                        if len(self.agent.memory) > self.sample_batch_size:
                            minibatch = random.sample(self.agent.memory, self.sample_batch_size)
                            self.agent.replay(minibatch)
                        break

            print("Episode finished!")
            print("************************")
        finally:
            print('******************************************************')
            print("Manual Override.... saving state.")
            # self.save_status(model, D, t, epsilon, loss_df, scores_df, actions_df, q_values_df)
            self.save_status(self.agent.model)
            print('- Finished Saving.')
            print('******************************************************')


def main():
    """Main executable for RL on dinosaur game."""
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
