import time
import greenabr
import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
import stable_baselines3
from stable_baselines3 import PPO
from greenabr.envs.videoplayer import stats
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

print(f"{gym.__version__=}")
print(f"{stable_baselines3.__version__=}")

n_timesteps = 100000
n_cpu = 16

policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=dict(pi=[256], vf=[256]))

env = make_vec_env('greenabr/greenabr-v0', n_envs=n_cpu)
# env = gym.make('greenabr/greenabr-v0')

model = PPO('MlpPolicy', env,
            verbose=0,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./logs/tensorboard_log/",
            learning_rate=3e-5,
            n_steps=64,  # 2048 1024 512 256 128 64 32 16 8
            batch_size=128,  # 256 128 64
            gamma=0.99, # 0.9999 0.99 0.98 0.95 0.9
            gae_lambda=0.95,  # 0.98 0.95 0.9 0.8
            n_epochs=10,  # 4 10 20
            clip_range=0.2,  # 0.2
            ent_coef=0.01  # 0.01 0.004 0.001 0.0
            )

# Use a separate environement for evaluation
eval_env = gym.make('greenabr/greenabr-v0')
eval_env = Monitor(eval_env)

checkpoint_callback = CheckpointCallback(
    save_freq=500000,
    save_path="./logs/PPO_2/",
    name_prefix="ppo_abr",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

eval_callback = EvalCallback(eval_env,
                             best_model_save_path="./logs/PPO_2/",
                             log_path="./logs/PPO_2/",
                             eval_freq=100,
                             deterministic=True,
                             render=False)

start_time = time.time()
model.learn(n_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True)
t = time.time() - start_time

print(f"Training took {time.strftime('%H:%M:%S', time.gmtime(t))}")

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10000)

print(f"[PPO AGENT] mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# Save the agent
model.save("/logs/debug_" + str(time.time()))
