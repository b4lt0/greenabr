import os
from keras.layers import Conv1D, Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

from joblib import dump, load

import gymnasium as gym
import greenabr

# Constants and training parameters

MODEL_NAME = 'baseline'
S_DIM = 7  # for VMAF included model
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 45.0  # number of chunks in the video 
M_IN_K = 1000.0
M_IN_N = 1000000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering 
SMOOTH_PENALTY = 1
RANDOM_SEED = 42
TRAIN_TRACES = '../cooked_traces/'  # network traces for training
PHONE_VMAF = pd.read_csv('vmaf_phone.csv')
POWER_ATTRIBUTES = pd.read_csv('../power_attributes.csv')  # video attributes used in power model estimations

# Max values for scaling power model attributes to be used in power model
BITRATE_MAX = 12000.0
FILE_SIZE_MAX = 1775324.0
QUALITY_MAX = 100.0
MOTION_MAX = 20.15
PIXEL_RATE_MAX = 2073600.0
POWER_MAX = 1690.0


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.int8)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        # store one hot encoding of actions, if appropriate
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


# NN for GreenABR agent
def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    model = Sequential([
        Dense(fc1_dims, input_shape=(input_dims,)),
        Activation('relu'),
        Dense(fc2_dims),
        Activation('relu'),
        Dense(128),
        Activation('relu'),
        Dense(64),
        Activation('relu'),
        Dense(n_actions)])

    model.compile(optimizer=Adam(lr=lr), loss='mse')
    return model


EPSILON_DECAY = 0.9995  # used for controlling exploration-exploitation trade off
MEMORY_SIZE = 500000  # size of replay buffer
TARGET_UPDATE_STEP = 100


class DDQNAgent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=EPSILON_DECAY, epsilon_end=0.01,
                 mem_size=MEMORY_SIZE, fname=MODEL_NAME, replace_target=TARGET_UPDATE_STEP):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname + '.h5'
        self.replace_target = replace_target
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions,
                                   discrete=True)
        self.q_eval = build_dqn(alpha, n_actions, input_dims, 256, 256)
        self.q_target = build_dqn(alpha, n_actions, input_dims, 256, 256)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)

            q_next = self.q_target.predict(new_state)
            q_eval = self.q_eval.predict(new_state)
            q_pred = self.q_eval.predict(state)

            max_actions = np.argmax(q_eval, axis=1)

            q_target = q_pred

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward + self.gamma * q_next[
                batch_index, max_actions.astype(int)] * done

            _ = self.q_eval.fit(state, q_target, verbose=0)

            if self.memory.mem_cntr % self.replace_target == 0:
                self.update_network_parameters()

    def update_network_parameters(self):
        self.q_target.set_weights(self.q_eval.get_weights())

    def decay_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    def save_model(self, it):
        self.q_eval.save('./savedModels/' + str(it) + '-' + self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)
        # if we are in evaluation mode we want to use the best weights for
        # q_target
        if self.epsilon == 0.0:
            self.update_network_parameters()


def main():
    all_cooked_time, all_cooked_bw, _ = load_trace(TRAIN_TRACES)
    input_dims = 7  # for VMAF included model 

    # env = Environment(all_cooked_time, all_cooked_bw, RANDOM_SEED)

    env = gym.make('greenabr/greenabr-v0', log_file=MODEL_NAME)

    ddqn_agent = DDQNAgent(alpha=0.0001, gamma=0.99, n_actions=6, epsilon=1.0, batch_size=64, input_dims=input_dims)
    n_games = 30000  # number of iterations to be used in training
    # ddqn_agent.load_model()
    ddqn_scores = []
    eps_history = []
    energy_cons = []
    data_cons = []
    qualities = []
    rebufferPens = []
    rebufferNums = []
    smoothPens = []
    smoothNums = []
    powerPens = []
    eps_iteration = []
    start_epoch = 0
    for i in range(n_games):
        done = False
        score = 0
        total_energy = 0
        total_data = 0
        last_bit_rate = 1
        total_quality = 0
        total_rebuffer_pen = 0
        total_rebuffer_time = 0
        total_smooth_pen = 0
        total_smooth_time = 0
        total_energy_penalty = 0

        observation = env.reset()
        while not done:
            action = ddqn_agent.choose_action(observation)

            observation_, reward, done, truncated, info = env.step(action)
            energy = info['estimated_energy']
            data = info['video_chunk_size']
            quality = info['quality']
            rebuffer_penalty = info['rebuffer_penalty']
            smooth_penalty = info['smooth_penalty']
            energy_penalty = info['energy_penalty']

            score += quality - rebuffer_penalty - smooth_penalty
            total_energy += energy
            total_data += data
            total_quality += quality
            total_rebuffer_pen = +rebuffer_penalty
            if rebuffer_penalty > 0:
                total_rebuffer_time += 1
            total_smooth_pen += smooth_penalty
            if smooth_penalty > 0:
                total_smooth_time += 1
            total_energy_penalty += energy_penalty
            ddqn_agent.remember(observation, action, reward, observation_, int(done))
            observation = observation_
            ddqn_agent.learn()
            ddqn_agent.decay_epsilon()
            last_bit_rate = action
            eps_history.append(ddqn_agent.epsilon)

        ddqn_scores.append(score)
        energy_cons.append(total_energy)
        data_cons.append(total_data)
        qualities.append(total_quality)
        rebufferPens.append(total_rebuffer_pen)
        rebufferNums.append(total_rebuffer_time)
        smoothPens.append(total_smooth_pen)
        smoothNums.append(total_smooth_time)
        powerPens.append(total_energy_penalty)
        eps_iteration.append(ddqn_agent.epsilon)
        avg_score = np.mean(ddqn_scores[max(0, i - 100):(i + 1)])
        print('episode: ', i, 'score: %.2f' % score,
              ' average score %.2f' % avg_score, 'epsilon: %.6f' % ddqn_agent.epsilon)

        if ((i + 1) % 1000) == 0 and i > 0:
            print('saving model')
            ddqn_agent.save_model(i + 1)

            all_stats = []
            for j in range(len(ddqn_scores)):
                all_stats.append(stats(i + 1, ddqn_scores[i], np.mean(ddqn_scores[max(0, j - 100):(j + 1)]),
                                       energy_cons[j], np.mean(energy_cons[max(0, j - 100):(j + 1)]),
                                       data_cons[j], np.mean(data_cons[max(0, j - 100):(j + 1)]),
                                       qualities[j], np.mean(qualities[max(0, j - 100):(j + 1)]),
                                       rebufferPens[j], rebufferNums[j],
                                       smoothPens[j], smoothNums[j],
                                       powerPens[j], np.mean(powerPens[max(0, j - 100):(j + 1)]),
                                       eps_iteration[j]))
            data = pd.DataFrame.from_records([s.to_dict() for s in all_stats])
            data.to_csv('Training_results.csv')


if __name__ == "__main__":
    main()
