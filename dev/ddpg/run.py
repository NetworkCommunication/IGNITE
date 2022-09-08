"""
-*- coding: utf-8 -*-
@author: Zijia Zhao
@Describtion:Gkmeans + ddpg
"""
from matplotlib import pyplot as plt
import pandas as pd
from env import ENV
from network import Agent
from GKmeans import Gkmeans
import numpy as np

cars = Gkmeans()
num_car = len(cars)
num_scar = int(num_car * 1 / 3)
num_tcar = num_car - num_scar
num_task = num_tcar
num_bs = 1

env = ENV(num_car, num_tcar, num_scar, num_task, num_bs, cars)
n_actions = 4
n_state = num_tcar * 3 + num_car + num_bs
MECSnet = Agent(alpha=0.0004, beta=0.004, input_dims=n_state,
                tau=0.01, env=env, batch_size=64, layer1_size=500,
                layer2_size=300, n_actions=n_actions)

score_record = []
score_record_step = []
count_record = []
count_record_step = []
episode_record = []
episode_record_step = []
cost_record = []
cost_record_step = []
print('服务车辆数量： {}'.format(num_scar))

for i in range(1000):
    score = 0
    obs = env.get_init_state(cars)
    done = False

    while not done:
        act = MECSnet.choose_action(obs)
        new_state, reward, done = env.step(act)
        MECSnet.remember(obs, act, reward, new_state, int(done))
        MECSnet.learn()
        score += reward
        obs = new_state

    episode_record.append(i)
    cost_record.append(-score)
    print('episode ', i, 'score %.2f' % score, "    wrong: ", env.count_wrong)
    count_record.append(1 - env.count_wrong / num_task)
    if i % 100 == 0:
        MECSnet.save_models()
        episode_record_step.append(i)
        cost_record_step.append(np.mean(cost_record))

df = pd.DataFrame({"Episode": episode_record_step, "Cost": cost_record_step}).set_index('Episode')
df.to_excel("excel_files/memorysize_10000_1.xlsx")

plt.figure()
x_data = range(len(cost_record_step))
plt.plot(x_data, cost_record_step)

plt.show()