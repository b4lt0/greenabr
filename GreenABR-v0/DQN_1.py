import time
import greenabr
import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
import stable_baselines3
from stable_baselines3 import DQN
from greenabr.envs.videoplayer import stats
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        obs = self.locals.get('new_obs')
        rew = self.locals.get('rewards')
        obs = obs.flatten()
        self.logger.record('REWARD', rew[0])
        self.logger.record('bit_rate', obs[0])
        self.logger.record('buffer_size', obs[1])
        self.logger.record('throughput', obs[2])
        self.logger.record('delay', obs[3])
        self.logger.record('remaining_chunks', obs[4])
        self.logger.record('VMAF', obs[6])
        return True

with tf.device('GPU'):
print(f"{gym.__version__=}")
print(f"{stable_baselines3.__version__=}")

EXP_NAME = 'DQN_1'

n_timesteps = 500000
n_cpu = 16

policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[256, 256, 128, 64])
env_kwargs = dict(log_file=EXP_NAME)
print(policy_kwargs)

env = make_vec_env('greenabr/greenabr-v0', env_kwargs=env_kwargs, n_envs=n_cpu)
# env = gym.make('greenabr/greenabr-v0')

model = DQN('MlpPolicy', env,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./logs/tensorboard_log/",
            verbose=0,
            learning_rate=0.0003,
            gamma=0.99,
            buffer_size=500000,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.01,
            exploration_fraction=0.9,
            target_update_interval=500,
            batch_size=64)

# Move the model to GPU
print('cuda device available:' + str(torch.cuda.is_available()))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Use a separate environement for evaluation
eval_env = gym.make('greenabr/greenabr-v0', log_file=EXP_NAME)
eval_env = Monitor(eval_env)

eval_callback = EvalCallback(eval_env,
                             best_model_save_path="./logs/" + EXP_NAME + "/",
                             log_path="./logs/" + EXP_NAME + "/",
                             eval_freq=1,
                             deterministic=True,
                             render=False)
tb_callback = TensorboardCallback()

start_time = time.time()
model.learn(n_timesteps,
            callback=[eval_callback, tb_callback],
            progress_bar=True)
t = time.time() - start_time

print(f"Training took {time.strftime('%H:%M:%S', time.gmtime(t))}")
