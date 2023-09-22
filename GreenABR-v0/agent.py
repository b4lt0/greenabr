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

n_timesteps = 2000000
n_cpu = 8

policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=dict(pi=[128, 128], vf=[128, 128]))

env = make_vec_env('greenabr/greenabr-v0', n_envs=n_cpu)

model = PPO('MlpPolicy', env,
             verbose=2, policy_kwargs=policy_kwargs,
             tensorboard_log="./tensorboard_log/")

# Use a separate environement for evaluation
eval_env = gym.make('greenabr/greenabr-v0')
eval_env = Monitor(eval_env)

checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path="./checkpoints_[128,128]/",
    name_prefix="ppo_abr",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

eval_callback = EvalCallback(eval_env,
                             best_model_save_path="./best_model_[128,128]/",
                             log_path="./best_model_[128,128]/",
                             eval_freq=50000,
                             deterministic=True,
                             render=False)

# Random Agent, before training
# mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
# print(f"[RANDOM AGENT] mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

start_time = time.time()
model.learn(n_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True)
t = time.time() - start_time

print(f"Training took {time.strftime('%H:%M:%S', time.gmtime(t))}")

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)

print(f"[PPO AGENT] mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# Save the agent
model.save("ppo_abr_128_" + str(n_timesteps))
