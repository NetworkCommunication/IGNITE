"""
-*- coding: utf-8 -*-
@author: Zijia Zhao
@Describtion:dqn run
"""
from matplotlib import pyplot as plt
import pandas as pd
from env import ENV
from dqn import DQN
import numpy as np

from dev.dqn import dqn

num_car = 15
num_task = 10
num_bs = 1

env = ENV(num_car, num_task, num_bs)
deep_dqn = DQN()

score_record = []
score_record_step = []
count_record = []
count_record_step = []
time_record = []
time_record_step = []
episode_record = []
episode_record_step = []

cost_record = []
cost_record_step = []
for i in range(5000):
    score = 0
    state = env.get_init_state()
    done = False
    while not done:
        act = deep_dqn.choose_action(state)
        new_state, reward, done = env.step(act)
        deep_dqn.store_transition(state, act, reward, new_state)
        score += reward
        state = new_state
        if deep_dqn.memory_counter > dqn.MEMORY_CAPACITY:
            deep_dqn.learn()
        print('reward is： {}'.format(reward))

    episode_record.append(i)
    cost_record.append(-score)
    score_record.append(score)
    print('episode ', i, 'score %.2f' % score, "    wrong: ", env.count_wrong)
    count_record.append(1 - env.count_wrong / num_task)
    if i % 20 == 0:
        episode_record_step.append(i)
        cost_record_step.append(np.mean(cost_record))
        score_record_step.append(np.mean(score_record))

df = pd.DataFrame({"Episode": episode_record_step, "Cost": cost_record_step}).set_index('Episode')
plt.figure()
x_data = range(len(cost_record))
plt.plot(x_data, cost_record)

plt.figure()
x_data = range(len(cost_record_step))
plt.plot(x_data, cost_record_step)

plt.show()
