
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump, load
import sys

import greenabr
import gymnasium as gym
import stable_baselines3
from stable_baselines3 import PPO
from greenabr.envs.videoplayer import stats


def setEvaluationFilesPath(video):
    return './evaluation/evaluationFiles/' + video + '/rep_6/'


if len(sys.argv) < 3:
    print(
        'Please run the script with one of the test videos (tos,bbb,doc). i.e. python evaluate.py tos model_file_name')
    quit()
else:
    VIDEO = sys.argv[1]
    if VIDEO not in ['tos', 'bbb', 'doc']:
        print('You must use one of the test videos, [tos,bbb,doc]')
        quit()
S_DIM = 7
A_DIM = 6
MODEL_NAME = sys.argv[2]

VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 44
M_IN_K = 1000.0
M_IN_N = 1000000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000

REWARD_FOLDER = './evaluation/rep_6/reward_logs/'
TEST_LOG_FOLDER = './evaluation/rep_6/test_results/'
TEST_TRACES = './evaluation/test_sim_traces/'
EVAL_FILE_PATH = setEvaluationFilesPath(VIDEO)
PHONE_VMAF = pd.read_csv(EVAL_FILE_PATH + 'vmaf_phone.csv')
REGULAR_VMAF = pd.read_csv(EVAL_FILE_PATH + 'vmaf_phone.csv')
POWER_ATTRIBUTES = pd.read_csv('evaluation/power_attributes.csv')
POWER_MES = pd.read_csv(EVAL_FILE_PATH + 'power_measurements.csv')

BITRATE_MAX = 12000.0
FILE_SIZE_MAX = 1775324.0
QUALITY_MAX = 100.0
MOTION_MAX = 20.15
PIXEL_RATE_MAX = 2073600.0
POWER_MAX = 1690.0


def load_trace(cooked_trace_folder=TEST_TRACES):
    cooked_files = os.listdir(cooked_trace_folder)
    if '.DS_Store' in cooked_files:
        cooked_files.remove('.DS_Store')
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for cooked_file in cooked_files:
        file_path = cooked_trace_folder + cooked_file
        cooked_time = []
        cooked_bw = []
        # print file_path
        with open(file_path, 'r') as f:
            for line in f:
                parse = line.split()
                if len(parse) > 0:
                    #                     print(parse[0], parse[1])
                    cooked_time.append(float(parse[0]))
                    cooked_bw.append(float(parse[1]))
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(cooked_file)

    return all_cooked_time, all_cooked_bw, all_file_names


# power model parameters for network power
p_alpha = 210
p_betha = 28

SEGMENT_SIZE = 4.0
power_threshold = 2500
byte_to_KB = 1000
KB_to_MB = 1000.0


def Estimate_Network_Power_Consumption(thr, chunk_file_size):
    return (chunk_file_size * (p_alpha * 1 / thr + p_betha))


MIN_ENERGY = 1000
NORMALIZATION_SCALAR = 1000
SCALING_FACTOR = -1.2


def Calculate_Energy_Penalty(energy_chunk):
    penalty = ((energy_chunk - MIN_ENERGY) / NORMALIZATION_SCALAR) ** 2 / SCALING_FACTOR
    return min(0, penalty)


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(7, input_dim=5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


import gymnasium as gym


class GreenABREvalEnv(gym.Wrapper):
    def __init__(self, env):
        super(GreenABREvalEnv, self).__init__(env)
        self.last_bit_rate = 0

    def step(self, bit_rate):
        (video_chunk_counter, delay, sleep_time, buffer_size, rebuf, video_chunk_size,
         next_video_chunk_sizes, end_of_video, video_chunk_remain) = self.get_video_chunk(bit_rate)

        throughput = float(video_chunk_size) / float(delay) / M_IN_K
        new_state = np.zeros(S_DIM)
        new_state[0] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))
        new_state[1] = buffer_size / BUFFER_NORM_FACTOR
        new_state[2] = throughput
        new_state[3] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR
        new_state[4] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

        log = np.zeros(6)
        log[0] = self.TOTAL_VIDEO_CHUNCK if video_chunk_counter == 0 else video_chunk_counter
        log[1] = delay
        log[2] = sleep_time
        log[3] = buffer_size
        log[4] = rebuf
        log[5] = video_chunk_size

        quality = (PHONE_VMAF['VMAF_' + str(bit_rate + 1)][log[0] - 1])
        new_state[6] = quality

        estimated_energy = (POWER_MES['P_' + str(bit_rate + 1)][log[0] - 1])
        new_state[5] = estimated_energy

        reward = 0
        quality = quality / 20.0
        quality_reward = quality

        if log[0] == 1:
            rebuffer_penalty = 0.0
        else:
            rebuffer_penalty = REBUF_PENALTY * rebuf

        if log[0] == 1:
            smooth_penalty = 0.0
        else:
            smooth_penalty = SMOOTH_PENALTY * np.abs((PHONE_VMAF['VMAF_' + str(bit_rate + 1)][log[0] - 1] / 20.0) - (
                    PHONE_VMAF['VMAF_' + str(self.last_bit_rate + 1)][log[0] - 2] / 20.0))

        reward = quality_reward - rebuffer_penalty - smooth_penalty

        energy_penalty = Calculate_Energy_Penalty(estimated_energy)

        self.last_bit_rate = bit_rate

        observation = np.array(new_state, dtype=np.float16)
        terminated = end_of_video
        truncated = False
        info = {'estimated_energy': estimated_energy, 'video_chunk_size': video_chunk_size, 'quality': quality,
                'rebuffer_penalty': rebuffer_penalty, 'smooth_penalty': smooth_penalty,
                'energy_penalty': energy_penalty, 'log': log}
        return observation, reward, terminated, truncated, info


def main():
    REWARD_MODEL = MODEL_NAME.split('.')[0]

    all_cooked_time, all_cooked_bw, all_file_names = load_trace(TEST_TRACES)

    input_dims = 7

    env = gym.make('greenabr/greenabr-v0', log_file='null', traces=TEST_TRACES)
    eval_env = GreenABREvalEnv(env)
    agent = PPO.load(MODEL_NAME, eval_env,
                     tensorboard_log="./logs/tensorboard_log/")

    #print(eval_env.get_wrapper_attr('trace_idx'))
    log_path = TEST_LOG_FOLDER + VIDEO + '/log_' + REWARD_MODEL + '_reward_' + all_file_names[env.trace_idx]
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, 'w')

    reward_log_path = './evaluation/rep_6/reward_logs/' + VIDEO + '/log_' + REWARD_MODEL + '_reward_' + all_file_names[
        env.trace_idx]
    os.makedirs(os.path.dirname(reward_log_path), exist_ok=True)
    reward_log_file = open(reward_log_path, 'w')


    trace_log_path = './evaluation/rep_6/reward_logs/' + VIDEO + "_log_" + REWARD_MODEL
    os.makedirs(os.path.dirname(trace_log_path), exist_ok=True)
    trace_log = open(trace_log_path, 'w')


    video_count = 0
    for video_count in range(len(all_file_names)):
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
        time_stamp = 0
        first_chunk = True
        log_path = TEST_LOG_FOLDER + VIDEO + '/log_' + REWARD_MODEL + '_reward_' + all_file_names[env.trace_idx]
        log_file = open(log_path, 'w')
        log_file.write('video_chunk' + '\t' +
                       'bitrate' + '\t' +
                       'buffer_size' + '\t' +
                       'rebuf' + '\t' +
                       'video_chunk_size' + '\t' +
                       'delay' + '\t' +
                       'phone_vmaf' + '\t' +
                       'regular_vmaf' + '\t' +
                       'energy' + '\t' +
                       'reward' + '\n'
                       )

        observation = env.reset()
        while not done:
            if first_chunk:
                bit_rate = 0
            else:
                action = agent.predict(observation)
                bit_rate = int(action[0])  # method predict() returns (action, next_state)
            print(str(bit_rate))
            observation_, reward, done, _, info = eval_env.step(bit_rate)

            energy = info['estimated_energy']
            data = info['video_chunk_size']
            quality = info['quality']
            rebuffer_penalty = info['rebuffer_penalty']
            smooth_penalty = info['smooth_penalty']
            energy_penalty = info['energy_penalty']
            log = info['log']

            first_chunk = False
            score += reward
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

            time_stamp += log[1]
            time_stamp += log[2]
            # print("the chunk counter is ", str(log[0]))
            log_file.write(str(log[0] - 1) + '\t' +
                           str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                           str(log[3]) + '\t' +
                           str(log[4]) + '\t' +
                           str(log[5]) + '\t' +
                           str(log[1]) + '\t' +
                           str(PHONE_VMAF['VMAF_' + str(bit_rate + 1)][log[0] - 1]) + '\t' +
                           str(REGULAR_VMAF['VMAF_' + str(bit_rate + 1)][log[0] - 1]) + '\t' +
                           str(energy) + '\t' +
                           str(reward) + '\n'
                           )

            log_file.flush()

            reward_log_file.write(str(log[0]) + '\t' +
                                  str(time_stamp) + '\t' +
                                  str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                                  str(quality) + '\t' +
                                  str(rebuffer_penalty) + '\t' +
                                  str(smooth_penalty) + '\t' +
                                  str(energy_penalty) + '\t' +
                                  str(PHONE_VMAF['VMAF_' + str(bit_rate + 1)][log[0] - 1]) + '\t' +
                                  str(REGULAR_VMAF['VMAF_' + str(bit_rate + 1)][log[0] - 1]) + '\t' +
                                  str(energy) + '\t' +
                                  str(reward) + '\n'
                                  )

            reward_log_file.flush()
            observation = observation_

        log_file.write('\n')
        log_file.close()
        reward_log_file.write('\n')
        reward_log_file.close()
        print('Completed trace ', all_file_names[env.trace_idx])
        print('Trace: ', video_count, 'score: %.2f' % score)
        trace_log.write(str(video_count + 1) + '\t' +
                        str(all_file_names[env.trace_idx] + '\t') +
                        str(score) + '\t' +
                        str(total_energy) + '\t' +
                        str(total_data) + '\t' +
                        str(total_quality) + '\t' +
                        str(total_rebuffer_pen) + '\t' +
                        str(total_rebuffer_time) + '\t' +
                        str(total_smooth_pen) + '\t' +
                        str(total_smooth_time) + '\t' +
                        str(total_energy_penalty) + '\n')
        trace_log.flush()
        video_count += 1
        reward_log_path = './evaluation/rep_6/reward_logs/' + VIDEO + '/log_' + REWARD_MODEL + '_reward_' + \
                          all_file_names[env.trace_idx]
        reward_log_file = open(reward_log_path, 'w')

    trace_log.write('\n')
    trace_log.close()


if __name__ == "__main__":
    main()
