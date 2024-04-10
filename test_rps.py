import gym
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPO
from rps_env import RockPaperScissorsEnv

def run_episode(env, policy):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = policy.compute_single_action(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward

if __name__ == "__main__":
    ray.init()

    # Initialize the environment and the trainer
    env = RockPaperScissorsEnv()
    trainer = PPO(env=RockPaperScissorsEnv, config={
        "env_config": {},  # Add any necessary configuration for your environment
    })

    # Load the trained model from a checkpoint
    checkpoint_path = "path/to/your/checkpoint"  # Update this path
    trainer.restore(checkpoint_path)

    # Now use the trained policy to run a test episode
    num_episodes = 1
    total_rewards = []
    for _ in range(num_episodes):
        total_reward = run_episode(env, trainer.get_policy())
        total_rewards.append(total_reward)
        print(f"Episode reward: {total_reward}")

    print(f"Average reward over {num_episodes} episodes: {np.mean(total_rewards)}")
