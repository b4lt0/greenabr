import time
import greenabr
import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
import stable_baselines3
from stable_baselines3 import A2C
from greenabr.envs.videoplayer import stats
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

print(f"{gym.__version__=}")
print(f"{stable_baselines3.__version__=}")

EXP_NAME = 'A2C_3'

n_timesteps = 300000
n_cpu = 16

policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128, 64, 32, 16],
                                   vf=[128, 128, 64, 64, 32]))
env_kwargs = dict(log_file=EXP_NAME)
print(policy_kwargs)

env = make_vec_env('greenabr/greenabr-v0', env_kwargs=env_kwargs, n_envs=n_cpu)
# env = gym.make('greenabr/greenabr-v0')

model = A2C('MlpPolicy', env,
            tensorboard_log="./logs/tensorboard_log/",
            verbose=0,
            learning_rate=7e-4,
            gamma=0.99,
            ent_coef=0.01,
            n_steps=8,
            vf_coef=0.4,
            max_grad_norm=0.5,
            gae_lambda=0.9,
            normalize_advantage=False
            )

# Use a separate environement for evaluation
eval_env = gym.make('greenabr/greenabr-v0', log_file=EXP_NAME)
eval_env = Monitor(eval_env)

eval_callback = EvalCallback(eval_env,
                             best_model_save_path="./logs/" + EXP_NAME + "/",
                             log_path="./logs/" + EXP_NAME + "/",
                             eval_freq=10,
                             deterministic=True,
                             render=False)
checkpoint_callback = CheckpointCallback(
                                        save_freq=10000,
                                        save_path="./logs/"+EXP_NAME,
                                        name_prefix=EXP_NAME,
                                        save_replay_buffer=True,
                                        save_vecnormalize=True,
                                    )

start_time = time.time()
model.learn(n_timesteps,
            callback=eval_callback,
            progress_bar=True)
t = time.time() - start_time

print(f"Training took {time.strftime('%H:%M:%S', time.gmtime(t))}")
