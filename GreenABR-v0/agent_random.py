import greenabr
import gc
import time
import numpy as np
import gymnasium as gym
import pandas as pd
from greenabr.envs.videoplayer import stats
from keras.layers import Dense
from keras.models import Sequential


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(7, input_dim=5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


env = gym.make('greenabr/greenabr-v0')
n_episodes = 100

# RANDOM AGENT
epsilon = 1
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

for i in range(n_episodes):
    start = time.process_time()
    gc.collect()
    terminated = False
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
    time_=0.0
    while not terminated:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        energy, data, quality, rebuffer_penalty, smooth_penalty, energy_penalty, t = info.values()
        time_+=t
        score += quality - rebuffer_penalty - smooth_penalty
        #         print("Action {0} , reward {1}, energy {2}, energy_pen {3}, reb_pen {4}, quality {5}, sm_pen {6}".format(action,reward,energy, energy_penalty, rebuffer_penalty, quality, smooth_penalty))
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
        #ddqn_agent.remember(observation, action, reward, observation_, int(terminated))
        #observation = observation_
        #ddqn_agent.learn()
        #ddqn_agent.decay_epsilon()
        #last_bit_rate = action
        #eps_history.append(ddqn_agent.epsilon)
        if terminated: break

    ddqn_scores.append(score)
    energy_cons.append(total_energy)
    data_cons.append(total_data)
    qualities.append(total_quality)
    rebufferPens.append(total_rebuffer_pen)
    rebufferNums.append(total_rebuffer_time)
    smoothPens.append(total_smooth_pen)
    smoothNums.append(total_smooth_time)
    powerPens.append(total_energy_penalty)
    #eps_iteration.append(ddqn_agent.epsilon)
    eps_iteration.append(epsilon)
    avg_score = np.mean(ddqn_scores[max(0, i - 100):(i + 1)])
    print('episode: ', i, 'score: %.2f' % score,
          ' average score %.2f' % avg_score,
          'epsilon: %.6f' % epsilon,
          'time: %.2fs' % (time.process_time() - start),
          'chunk: %.2fs' % time_)



    if ((i + 1) % 1000) == 0 and i > 0:
        print('saving model')
        #ddqn_agent.save_model(i + 1)

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
        data.to_csv('Training_results_'+env.get_name()+'.csv')

