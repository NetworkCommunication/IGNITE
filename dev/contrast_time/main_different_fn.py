"""
@File: main_different_fn.py
@author: Zijia Zhao
@Describtion: Comparative experiments of computing power of different vehicles
"""
from matplotlib import pyplot as plt
import pandas as pd
from env_gk_ddpg_contrast import ENV_gk_ddpg
from env_gk_dqn_contrast import ENV_gk_dqn
from env_gk_ddpg_local_contrast import ENV_gk_ddpg_local
from network import Agent
import numpy as np
from dqn import DQN
from dev.contrast import dqn

num_car = 15
num_scar = int(num_car * 1 / 3)
num_tcar = num_car - num_scar
num_task = num_tcar
num_bs = 1

def gk_ddpg(episode, filename, fn_car, fn_bs, cn, step):
    env = ENV_gk_ddpg(num_car, num_tcar, num_scar, num_task, num_bs, fn_car, fn_bs, cn)
    n_actions = 4
    n_state = num_tcar * 3 + num_car + num_bs
    MECSnet = Agent(alpha=0.0004, beta=0.004, input_dims=n_state,
                    tau=0.01, env=env, batch_size=64, layer1_size=500,
                    layer2_size=300, n_actions=n_actions)

    episode_record = []
    episode_record_step = []
    cost_record = []
    cost_record_step = []
    time_record = []
    time_record_step = []
    for i in range(episode):
        score = 0
        obs = env.get_init_state()
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
        time_record.append(env.time)
        print('episode ', i, 'score %.2f' % score, "    wrong: ", env.count_wrong)
        if i % step == 0:
            MECSnet.save_models()
            episode_record_step.append(i)
            cost_record_step.append(np.mean(cost_record))
            time_record_step.append(np.mean(time_record))
    df_cost = pd.DataFrame({"Episode": episode_record_step, "Time": time_record_step}).set_index('Episode')
    df_cost.to_excel("gk_ddpg/car_fn/{}.xlsx".format(filename + '_gk_ddpg'))

    plt.figure()
    x_data = range(len(time_record_step))
    plt.plot(x_data, time_record_step)

    plt.show()

def gk_dqn(episode, filename, fn_car, fn_bs, cn, step):
    env = ENV_gk_dqn(num_car, num_task, num_bs, fn_car, fn_bs, cn)

    deep_dqn = DQN()

    episode_record = []
    episode_record_step = []
    cost_record = []
    cost_record_step = []
    time_record = []
    time_record_step = []
    for i in range(episode):
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

        episode_record.append(i)
        cost_record.append(-score)
        time_record.append(env.time)
        print('episode ', i, 'score %.2f' % score, "    wrong: ", env.count_wrong)
        if i % step == 0:
            episode_record_step.append(i)
            cost_record_step.append(np.mean(cost_record))
            time_record_step.append(np.mean(time_record))
    df_cost = pd.DataFrame({"Episode": episode_record_step, "Time": time_record_step}).set_index('Episode')
    df_cost.to_excel("gk_dqn/car_fn/{}.xlsx".format(filename + '_gk_dqn'))

    plt.figure()
    x_data = range(len(time_record_step))
    plt.plot(x_data, time_record_step)
    plt.show()

def gk_ddpg_local(episode, filename, fn_car, fn_bs, cn, step):
    env = ENV_gk_ddpg_local(num_car, num_tcar, num_scar, num_task, num_bs, fn_car, fn_bs, cn)
    n_actions = 4
    n_state = num_tcar * 3 + num_car + num_bs
    MECSnet = Agent(alpha=0.0004, beta=0.004, input_dims=n_state,
                    tau=0.01, env=env, batch_size=64, layer1_size=500,
                    layer2_size=300, n_actions=n_actions)

    episode_record = []
    episode_record_step = []
    cost_record = []
    cost_record_step = []
    time_record = []
    time_record_step = []

    for i in range(episode):
        score = 0
        obs = env.get_init_state()
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
        time_record.append(env.time)
        print('episode ', i, 'score %.2f' % score, "    wrong: ", env.count_wrong)
        if i % step == 0:
            MECSnet.save_models()
            episode_record_step.append(i)
            cost_record_step.append(np.mean(cost_record))
            time_record_step.append(np.mean(time_record))
    df_cost = pd.DataFrame({"Episode": episode_record_step, "Time": time_record_step}).set_index('Episode')
    df_cost.to_excel("gk_ddpg_local/car_fn/{}.xlsx".format(filename + '_gk_ddpg_local'))

    plt.figure()
    x_data = range(len(time_record_step))
    plt.plot(x_data, time_record_step)

    plt.show()


if __name__ == '__main__':
    episode = 50
    step = 25
    round = 1
    fn = 0.5

    fn_bs = 0
    cn = 0
    fn_car = fn * 1e8
    filename = str(round) + '_' + str(fn) + '_car_fn'
    # gkmeans+ddpg
    gk_ddpg(episode, filename, fn_car, fn_bs, cn, step)
    # gkmeans+dqn
    gk_dqn(episode, filename, fn_car, fn_bs, cn, step)
    # gkmeans+ddpg+local
    gk_ddpg_local(episode, filename, fn_car, fn_bs, cn, step)
