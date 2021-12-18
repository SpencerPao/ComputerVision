import gym
import gym_dinorun
# from WebInterface import WebInterface


def main():
    env = gym.make("DinoRun-v0")
    env.reset()
    for _ in range(1000):  # Number of "steps": Make steps large for training.
        # env.render()
        env.step(env.action_space.sample())  # take a random action
    print("Closing Environment")
    env.close()


if __name__ == '__main__':
    main()
