import sys
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

if len(sys.argv) < 3:
    print(
        'Please run the script with the model name and the number of steps. i.e. python continue.py A2C_1 10000')
    quit()
else:
    EXP_NAME = sys.argv[1]
    N_TIMESTEPS = int(sys.argv[2])

N_CPU = 16

env_kwargs = dict(log_file=EXP_NAME)

env = make_vec_env('greenabr/greenabr-v0', env_kwargs=env_kwargs, n_envs=N_CPU)

eval_env = gym.make('greenabr/greenabr-v0', log_file=EXP_NAME)
eval_env = Monitor(eval_env)

eval_callback = EvalCallback(eval_env,
                             best_model_save_path="./logs/"+EXP_NAME+"/",
                             log_path="./logs/"+EXP_NAME+"/",
                             eval_freq=10,
                             deterministic=True,
                             render=False)

policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=dict(pi=[128, 64, 32, 16], vf=[128, 128, 64, 64, 32]))

model = PPO.load(path='./logs/'+EXP_NAME+'/best_model.zip',
                 env=env,
                 verbose=0
                 )

start_time = time.time()
model.learn(total_timesteps=N_TIMESTEPS,
            callback=eval_callback,
            progress_bar=True,
            log_interval=1,
            tb_log_name=EXP_NAME,
            reset_num_timesteps=False)
t = time.time() - start_time

print(f"Training took {time.strftime('%H:%M:%S', time.gmtime(t))}")

