#!/usr/bin/env python3

import gym
import crafter

# env = gym.make("CrafterReward-v1")  # Or CrafterNoReward-v1
# env = crafter.Recorder(
    # env,
    # "./path/to/logdir",
    # save_stats=True,
    # save_video=True,
    # save_episode=False,
# )

# obs = env.reset()
# done = False
# while not done:
    # action = env.action_space.sample()
    # obs, reward, done, info = env.step(action)


# Same as above, but using PPO


from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

env = gym.make("CrafterReward-v1")  # Or CrafterNoReward-v1
env = DummyVecEnv([lambda: env])

model = PPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=1e5)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")

