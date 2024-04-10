import gymnasium as gym
from gym import spaces
import numpy as np
import random
from ray.rllib.env import ExternalEnv
import ray
from ray import tune


# Step 1: Create a Gym Environment for Rock, Paper, Scissors
class RockPaperScissorsEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(3)  # 0: Rock, 1: Paper, 2: Scissors
        self.observation_space = spaces.Discrete(3)  # Dummy observation space

    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"
        computer_action = random.choice([0, 1, 2])
        reward = self.determine_reward(action, computer_action)
        done = True  # Each game is only one step
        info = {"computer_action": computer_action}
        return np.array([0]), reward, done, info

    def reset(self):
        return np.array([0])  # Return a dummy observation

    def render(self, mode="human"):
        pass  # Optional for visualization

    def determine_reward(self, user_action, computer_action):
        if user_action == computer_action:
            return 0  # Draw
        elif (user_action == 0 and computer_action == 2) or \
             (user_action == 1 and computer_action == 0) or \
             (user_action == 2 and computer_action == 1):
            return 1  # Win
        else:
            return -1  # Lose

# Step 2: Integrate with RLlib's External Environment (this is a sketch; customization needed)
class RPSExternalEnv(ExternalEnv):
    def __init__(self, env_config):
        self.env = RockPaperScissorsEnv()
        super().__init__(action_space=self.env.action_space, observation_space=self.env.observation_space)

    def run(self):
        obs = self.env.reset()
        done = False
        while not done:
            action = self.get_action(obs, done)
            obs, reward, done, info = self.env.step(action)
            self.send_observation(obs, reward, done, info)
        self.send_done()

# Example on how to use it with RLlib (you would typically run this in a script or notebook)
# from ray import tune
# tune.run(
#     "PPO",
#     config={
#         "env": RPSExternalEnv,
#         "num_workers:": 1,
#         # Add other necessary config options here, such as "num_workers"
#     }
# )
ray.init()

# Configure and run the training
tune.run(
    "PPO",  # You can use other algorithms as well, such as "A3C", "DQN", etc.
    config={
        "env": RPSExternalEnv,  # Your custom environment
        "num_workers": 1,  # Adjust based on your system's capabilities
        "env_config": {},  # Any configuration specific to your environment
        "framework": "torch",  # Or "torch" if you prefer PyTorch
    },
    stop={
        "training_iteration": 10  # Adjust the number of training iterations
    }
)