import gymnasium as gym
from gym import spaces
from gymnasium.spaces import Box, Discrete
import numpy as np
import random

class RockPaperScissorsEnv(gym.Env):
    def __init__(self, env_config=None):
        super(RockPaperScissorsEnv, self).__init__()
        self.action_space = Box(low=0, high=2, dtype=np.int_)  # 0: Rock, 1: Paper, 2: Scissors
        self.observation_space = Box(low=0, high=2, dtype=np.int_) # Dummy observation, not used

    def reset(self):
        return np.array([0])  # Reset the state of the environment to an initial state

    def step(self, action):
        assert self.action_space.contains(action)
        computer_action = random.choice([0, 1, 2])
        if action == computer_action:
            reward = 0  # Draw
        elif (action == 0 and computer_action == 2) or \
             (action == 1 and computer_action == 0) or \
             (action == 2 and computer_action == 1):
            reward = 1  # Win
        else:
            reward = -1  # Lose
        done = True  # Single step per episode for simplicity
        return np.array([0]), reward, done, {}


from ray.rllib.env import ExternalEnv

class RPSExternalEnv(ExternalEnv):
    def __init__(self, env_config={}):
        self.env = RockPaperScissorsEnv(env_config)
        super().__init__(self.env.action_space, self.env.observation_space)

    def run(self):
        while True:
            action = self.get_action()  # Assuming you have a method to get actions
            obs, reward, done, info = self.env.step(action)
            self.send_observation(obs, reward, done, info)
            if done:
                self.end_episode(reward, obs)
