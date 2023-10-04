import gymnasium as gym


class GreenABREvalEnv(gym.Wrapper):
    def __init__(self, env):
        super(GreenABREvalEnv, self).__init__(env)

    def step(self, action):
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
                        PHONE_VMAF['VMAF_' + str(last_bitrate + 1)][log[0] - 2] / 20.0))

        reward = quality_reward - rebuffer_penalty - smooth_penalty

        energy_penalty = Calculate_Energy_Penalty(estimated_energy)

        self.last_bit_rate = bit_rate

        observation = np.array(new_state, dtype=np.float16)
        terminated = end_of_video
        truncated = False
        info = {'estimated_energy': estimated_energy, 'video_chunk_size': video_chunk_size, 'quality': quality,
                'rebuffer_penalty': rebuffer_penalty, 'smooth_penalty': smooth_penalty,
                'energy_penalty': energy_penalty, 'log':log}
        return observation, reward, terminated, truncated, info