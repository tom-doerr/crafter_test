#!/usr/bin/env python3

import gym
import crafter
import visdom 
import numpy as np

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



vis = visdom.Visdom()

def callback_function(o1, o2):
    # o1: {'self': <stable_baselines3.ppo.ppo.PPO object at 0x7f7c00317100>, 'total_timesteps': 100000.0, 'callback': <stable_baselines3.common.callbacks.ConvertCallback object at 0x7f7c9d4d8670>, 'log_interval': 1, 'eval_env': None, 'eval_freq': -1, 'n_eval_episodes': 5, 'tb_log_name': 'PPO', 'eval_log_path': None, 'reset_num_timesteps': True, 'iteration': 0, 'env': <stable_baselines3.common.vec_env.vec_transpose.VecTransposeImage object at 0x7f7c003170a0>, 'rollout_buffer': <stable_baselines3.common.buffers.RolloutBuffer object at 0x7f7c00317190>, 'n_rollout_steps': 2048, 'n_steps': 7, 'obs_tensor': tensor([[[[59, 50, 50,  ..., 50, 50,  0],
          # [59, 50, 50,  ..., 59, 50,  0],
          # [59, 50, 50,  ..., 59, 50,  0],
          # ...,
          # [ 0,  0,  0,  ...,  0,  0,  0],
          # [ 0,  0,  0,  ...,  0,  0,  0],
          # [ 0,  0,  0,  ...,  0,  0,  0]],

         # [[59, 50, 50,  ..., 50, 50,  0],
          # [59, 50, 50,  ..., 59, 50,  0],
          # [59, 50, 50,  ..., 59, 50,  0],
          # ...,
          # [ 0,  0,  0,  ...,  0,  0,  0],
          # [ 0,  0,  0,  ...,  0,  0,  0],
          # [ 0,  0,  0,  ...,  0,  0,  0]],

         # [[67, 58, 58,  ..., 58, 58,  0],
          # [67, 58, 58,  ..., 67, 58,  0],
          # [67, 58, 58,  ..., 67, 58,  0],
          # ...,
          # [ 0,  0,  0,  ...,  0,  0,  0],
          # [ 0,  0,  0,  ...,  0,  0,  0],
          # [ 0,  0,  0,  ...,  0,  0,  0]]]], device='cuda:0',
       # dtype=torch.uint8), 'actions': array([16]), 'values': tensor([[1.5782]], device='cuda:0'), 'log_probs': tensor([-2.5246], device='cuda:0'), 'clipped_actions': array([16]), 'new_obs': array([[[[59, 50, 50, ..., 50, 50,  0],
         # [59, 50, 50, ..., 59, 50,  0],
         # [59, 50, 50, ..., 59, 50,  0],
         # ...,
         # [ 0,  0,  0, ...,  0,  0,  0],
         # [ 0,  0,  0, ...,  0,  0,  0],
         # [ 0,  0,  0, ...,  0,  0,  0]],

        # [[59, 50, 50, ..., 50, 50,  0],
         # [59, 50, 50, ..., 59, 50,  0],
         # [59, 50, 50, ..., 59, 50,  0],
         # ...,
         # [ 0,  0,  0, ...,  0,  0,  0],
         # [ 0,  0,  0, ...,  0,  0,  0],
         # [ 0,  0,  0, ...,  0,  0,  0]],

        # [[67, 58, 58, ..., 58, 58,  0],
         # [67, 58, 58, ..., 67, 58,  0],
         # [67, 58, 58, ..., 67, 58,  0],
         # ...,
         # [ 0,  0,  0, ...,  0,  0,  0],
         # [ 0,  0,  0, ...,  0,  0,  0],
         # [ 0,  0,  0, ...,  0,  0,  0]]]], dtype=uint8), 'rewards': array([0.], dtype=float32), 'dones': array([False]), 'infos': [{'inventory': {'health': 9, 'food': 8, 'drink': 7, 'energy': 8, 'sapling': 0, 'wood': 0, 'stone': 0, 'coal': 0, 'iron': 0, 'diamond': 0, 'wood_pickaxe': 0, 'stone_pickaxe': 0, 'iron_pickaxe': 0, 'wood_sword': 0, 'stone_sword': 0, 'iron_sword': 0}, 'achievements': {'collect_coal': 0, 'collect_diamond': 0, 'collect_drink': 0, 'collect_iron': 0, 'collect_sapling': 0, 'collect_stone': 0, 'collect_wood': 0, 'defeat_skeleton': 0, 'defeat_zombie': 0, 'eat_cow': 0, 'eat_plant': 0, 'make_iron_pickaxe': 0, 'make_iron_sword': 0, 'make_stone_pickaxe': 0, 'make_stone_sword': 0, 'make_wood_pickaxe': 0, 'make_wood_sword': 0, 'place_furnace': 0, 'place_plant': 0, 'place_stone': 0, 'place_table': 0, 'wake_up': 0}, 'discount': 1.0, 'semantic': array([[ 1,  1,  5, ...,  4,  3,  4],
       # [ 1,  1,  1, ..., 10,  3,  4],
       # [ 1,  1,  1, ...,  3,  3,  4],
       # ...,
       # [ 3,  3,  3, ...,  2,  2,  2],
       # [ 4,  2,  1, ...,  2,  2,  2],
       # [ 6,  2,  1, ...,  5,  5,  2]], dtype=uint8), 'player_pos': array([30, 33]), 'reward': 0.0}]}


    # Get the reward from the abouve output.
    reward = o1['rewards']
    # Plot it using visdom.
    if o1['n_steps'] % 100 == 0:
        vis.line(X=np.array([o1['n_steps']]), Y=np.array([reward]), win='reward', update='append', opts={'title': 'Reward'})

# Same as above, but using PPO


from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

env = gym.make("CrafterReward-v1")  # Or CrafterNoReward-v1
env = DummyVecEnv([lambda: env])

model = PPO(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=1e5)

# Same as above, but show reward
model.learn(total_timesteps=1e5, callback=callback_function)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")

