import os
import time

import numpy as np
import pandas as pd
from joblib import load

import gymnasium as gym
from gymnasium import spaces

from keras.layers import Dense
from keras.models import Sequential

from memory_profiler import profile

print(os.getcwd())

# Constants and training parameters

MODEL_NAME = 'GreenABR_random_agent'
S_DIM = 7  # for VMAF included model
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 45.0  # number of chunks in the video
M_IN_K = 1000.0
M_IN_N = 1000000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering
SMOOTH_PENALTY = 1
RANDOM_SEED = 42
TRAIN_TRACES = './greenabr/cooked_traces/'  # network traces for training
PHONE_VMAF = pd.read_csv('./greenabr/envs/vmaf_phone.csv')
POWER_ATTRIBUTES = pd.read_csv('./greenabr/power_attributes.csv')  # video attributes used in power model estimations

# Max values for scaling power model attributes to be used in power model
BITRATE_MAX = 12000.0
FILE_SIZE_MAX = 1775324.0
QUALITY_MAX = 100.0
MOTION_MAX = 20.15
PIXEL_RATE_MAX = 2073600.0
POWER_MAX = 1690.0


#  keeps the training logs
class stats:
    def __init__(self, iteration, score, avgScore, power, avgPower, data, avgData, quality, avgQuality, rebufferPen,
                 numRebuffer, smoothPen, numSmooth, powerPen, avgPowerPen, eps_iter):
        self.iteration = iteration
        self.score = score
        self.avgScore = avgScore
        self.power = power
        self.avgPower = avgPower
        self.data = data
        self.avgData = avgData
        self.quality = quality
        self.avgQuality = avgQuality
        self.rebufferPen = rebufferPen
        self.numRebuffer = numRebuffer
        self.smoothPen = smoothPen
        self.numSmooth = numSmooth
        self.powerPen = powerPen
        self.avgPowerPen = avgPowerPen
        self.epsilon = eps_iter

    def to_dict(self):
        return {
            'iteration': self.iteration,
            'score': self.score,
            'avgScore': self.avgScore,
            'power': self.power,
            'avgPower': self.avgPower,
            'data': self.data,
            'avgData': self.avgData,
            'quality': self.quality,
            'avg_quality': self.avgQuality,
            'rebufferPen': self.rebufferPen,
            'numrebuffer': self.numRebuffer,
            'smoothPen': self.smoothPen,
            'numSmooth': self.numSmooth,
            'powerPen': self.powerPen,
            'avgPowerPen': self.avgPowerPen,
            'epsilon': self.epsilon

        }


def load_trace(cooked_trace_folder=TRAIN_TRACES):
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


# parameters to be used in network power model
p_alpha = 210
p_betha = 28

SEGMENT_SIZE = 4.0
power_threshold = 2500
byte_to_KB = 1000
KB_to_MB = 1000.0


def Estimate_Network_Power_Consumption(thr, chunk_file_size):
    return (chunk_file_size * (p_alpha * 1 / thr + p_betha))


# constants used in energy consumption penalty
MIN_ENERGY = 1000
NORMALIZATION_SCALAR = 1000


def Calculate_Energy_Penalty(energy_chunk):
    penalty = (energy_chunk - MIN_ENERGY) / NORMALIZATION_SCALAR
    if energy_chunk > 3000:
        penalty = penalty + 2 ** ((energy_chunk - 3000) / 1000)
    penalty = penalty * -1
    return penalty


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(7, input_dim=5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


class GreenABREnv(gym.Env):
    def __init__(self, random_seed=RANDOM_SEED):
        all_cooked_time, all_cooked_bw, _ = load_trace(TRAIN_TRACES)
        self.MODEL_NAME = MODEL_NAME
        self.MILLISECONDS_IN_SECOND = 1000.0
        self.B_IN_MB = 1000000.0
        self.BITS_IN_BYTE = 8.0
        self.RANDOM_SEED = 42
        self.VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
        self.BITRATE_LEVELS = 6
        self.TOTAL_VIDEO_CHUNCK = 45
        self.BUFFER_THRESH = 60000.0  # millisec, max buffer limit
        self.DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
        self.PACKET_PAYLOAD_PORTION = 0.95
        self.LINK_RTT = 80  # millisec
        self.PACKET_SIZE = 1500  # bytes
        self.NOISE_LOW = 0.9
        self.NOISE_HIGH = 1.1
        self.VIDEO_SIZE_FILE = './greenabr/envs/video_size_'
        assert len(all_cooked_time) == len(all_cooked_bw)

        self.local_power_model = load('./greenabr/power_model.pkl')  # loads the power pre-trained power model
        #self.local_power_model = load(
            #'/Users/andreabalillo/PycharmProjects/greenabr/GreenABR-v0/greenabr/power_model.pkl')  # loads the power pre-trained power model


        np.random.seed(random_seed)

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw

        self.video_chunk_counter = 0
        self.buffer_size = 0
        self.last_bit_rate = 1

        self.action_space = spaces.Discrete(len(VIDEO_BIT_RATE))
        # [bitrate, buffer_size, throughput, download_time, remaining_chunks, energy, QoE]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float16),
            high=np.array([1.0, 6.0, 1.0, 6.0, 1.0, 3000.0, 100.0], dtype=np.float16),
            shape=(S_DIM,),
            dtype=np.float16
        )

        # pick a random trace file
        self.trace_idx = np.random.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.video_size = {}  # in bytes
        for bitrate in range(self.BITRATE_LEVELS):
            self.video_size[bitrate] = []
            with open(self.VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

    def get_name(self):
        return self.MODEL_NAME

    def reset(self, seed=None, options=None):
        self.video_chunk_counter = 0
        self.buffer_size = 0

        # pick a random trace file
        self.trace_idx = np.random.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.video_size = {}  # in bytes
        for bitrate in range(self.BITRATE_LEVELS):
            self.video_size[bitrate] = []
            with open(self.VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

        info = {'estimated_energy': 0.0, 'video_chunk_size': 0.0, 'quality': 0.0,
                'rebuffer_penalty': 0.0, 'smooth_penalty': 0.0,
                'energy_penalty': 0.0}

        return np.zeros(S_DIM, dtype=np.float16), info

    def get_video_chunk(self, quality):

        assert quality >= 0
        assert quality < self.BITRATE_LEVELS

        video_chunk_size = self.video_size[quality][self.video_chunk_counter]

        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes

        while True:  # download video chunk over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr] * self.B_IN_MB / self.BITS_IN_BYTE
            duration = self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time

            packet_payload = throughput * duration * self.PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:
                fractional_time = (
                                          video_chunk_size - video_chunk_counter_sent) / throughput / self.PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                assert (self.last_mahimahi_time <= self.cooked_time[self.mahimahi_ptr])
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0

        delay *= self.MILLISECONDS_IN_SECOND
        delay += self.LINK_RTT

        # add a multiplicative noise to the delay
        delay *= np.random.uniform(self.NOISE_LOW, self.NOISE_HIGH)

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new chunk
        self.buffer_size += self.VIDEO_CHUNCK_LEN

        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > self.BUFFER_THRESH:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - self.BUFFER_THRESH
            sleep_time = np.ceil(drain_buffer_time / self.DRAIN_BUFFER_SLEEP_TIME) * self.DRAIN_BUFFER_SLEEP_TIME
            self.buffer_size -= sleep_time

            while True:
                duration = self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time
                if duration > sleep_time / self.MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time += sleep_time / self.MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * self.MILLISECONDS_IN_SECOND
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0

            # the "last buffer size" return to the controller
            # Note: in old version of dash the lowest buffer is 0.
            # In the new version the buffer always have at least
            # one chunk of video
        return_buffer_size = self.buffer_size

        self.video_chunk_counter += 1
        video_chunk_remain = self.TOTAL_VIDEO_CHUNCK - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= self.TOTAL_VIDEO_CHUNCK:
            end_of_video = True
            self.buffer_size = 0
            self.video_chunk_counter = 0

            # pick a random trace file
            self.trace_idx = np.random.randint(len(self.all_cooked_time))
            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            # randomize the start point of the video
            # note: trace file starts with time 0
            self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        next_video_chunk_sizes = []
        for i in range(self.BITRATE_LEVELS):
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])

        return self.video_chunk_counter, delay, sleep_time, return_buffer_size / self.MILLISECONDS_IN_SECOND, rebuf / self.MILLISECONDS_IN_SECOND, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain

    # normalize the attributes for power model
    def normalize_parameters(self, bitrate, t_sec):
        time = self.video_chunk_counter * SEGMENT_SIZE + t_sec
        b_n = VIDEO_BIT_RATE[bitrate] / BITRATE_MAX
        d = POWER_ATTRIBUTES[(POWER_ATTRIBUTES['Bitrate'] == VIDEO_BIT_RATE[bitrate]) &
                             (POWER_ATTRIBUTES['Time'] == time)]
        f_n = d['FileSize'] / FILE_SIZE_MAX
        q_n = d['Quality'] / QUALITY_MAX
        m_n = d['Motion'] / MOTION_MAX
        p_n = d['PixelRate'] / PIXEL_RATE_MAX
        return np.reshape(np.array([b_n, f_n.iloc[0], q_n.iloc[0], m_n.iloc[0], p_n.iloc[0]]), (1, 5))

    def calculate_local_energy(self, bitrate):
        total_energy = 0
        t = 0
        for i in range(int(SEGMENT_SIZE)):
            pars = self.normalize_parameters(bitrate, i)

            #TODO aaaaaaaaaaaaaaaaaaaaaaaa
            start = time.process_time()
            pred = self.local_power_model.predict_on_batch(pars)
            t += (time.process_time() - start)

            power = pred[0] * POWER_MAX  # equal to energy as it is for 1 second
            total_energy += power
        return total_energy, t

    # corresponds to an ABR selection at every video chunk
    def step(self, bit_rate):
        video_chunk_counter, delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain = self.get_video_chunk(
            bit_rate)
        throughput = float(video_chunk_size) / float(delay) / M_IN_K  # convert to MBps

        #         print(throughput)
        if video_chunk_counter == 0:
            video_chunk_counter = self.TOTAL_VIDEO_CHUNCK
        new_state = np.zeros(S_DIM)
        new_state[0] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))
        new_state[1] = buffer_size / BUFFER_NORM_FACTOR
        new_state[2] = throughput  # original throughput
        new_state[3] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR
        new_state[4] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

        network_energy = Estimate_Network_Power_Consumption(throughput * self.BITS_IN_BYTE, video_chunk_size) / M_IN_N

        # TODO make this more efficient
        local_play_energy, t = self.calculate_local_energy(bit_rate)

        estimated_energy = network_energy + local_play_energy
        new_state[5] = estimated_energy
        quality = (PHONE_VMAF['VMAF_' + str(bit_rate + 1)][video_chunk_counter - 1])
        new_state[6] = quality  # last VMAF value needs to be commented for regular model

        reward = 0
        quality = quality / 20.0  # normalizing quality
        quality_reward = quality
        if quality > 3:
            quality_reward = quality + 2 ** (quality - 3)  # nonlinear quality component to prefer higher qualities
        rebuffer_penalty = REBUF_PENALTY * rebuf
        if video_chunk_counter == 1:
            smooth_penalty = 0.0
        else:
            smooth_penalty = SMOOTH_PENALTY * np.abs(
                (PHONE_VMAF['VMAF_' + str(bit_rate + 1)][video_chunk_counter - 1] / 20.0) - (
                        PHONE_VMAF['VMAF_' + str(self.last_bit_rate + 1)][video_chunk_counter - 2] / 20.0))
        reward = quality_reward - rebuffer_penalty - smooth_penalty
        energy_penalty = Calculate_Energy_Penalty(estimated_energy)
        reward = reward + float(energy_penalty)

        self.last_bit_rate = bit_rate

        #         print('Quality: {} Rebuffer Pen {} Smoothness Penalty {} Energy Penalty {} Estimated Energy {}'.format(quality_reward, rebuffer_penalty, smooth_penalty, energy_penalty, estimated_energy))
        # return new_state, reward, end_of_video, estimated_energy, video_chunk_size, quality, rebuffer_penalty, smooth_penalty, energy_penalty
        observation = np.array(new_state, dtype=np.float16)
        terminated = end_of_video
        truncated = False
        info = {'estimated_energy': estimated_energy, 'video_chunk_size': video_chunk_size, 'quality': quality,
                'rebuffer_penalty': rebuffer_penalty, 'smooth_penalty': smooth_penalty,
                'energy_penalty': energy_penalty, 'time': t}
        return observation, reward, terminated, truncated, info

