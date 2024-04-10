import gymnasium as gym
from gym import spaces
from gymnasium.spaces import Box, Discrete
import numpy as np
import random
import threading


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
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.action = None

    def start(self):
        """Starts the environment thread."""
        self.thread.start()

    def run(self):
        """The main loop that interacts with the environment asynchronously."""
        while True:
            # Here you would implement the logic to wait for an action,
            # step the environment, and then send back observations.
            # This is a simplified placeholder.
            if self.action is not None:
                observation, reward, done, info = self.env.step(self.action)
                self.send_observation(observation, reward, done, info)
                if done:
                    self.end_episode(reward, observation)
                self.action = None  # Reset action after processing

    # Placeholder for setting actions. In a real scenario, this would
    # be more complex, handling synchronization between threads.
    def set_action(self, action):
        """Sets the action to be executed in the environment."""
        self.action = action
        return "yo"



if __name__ == "__main__":
    # Initialize and start the environment thread
    env = RPSExternalEnv()
    env.start()

    # Example of setting an action
    # In a real application, actions would typically come from an RLlib policy or similar
    info =env.set_action(0)  # Assuming 0 is a valid action in your environment
    print(info)
    # Your application logic here