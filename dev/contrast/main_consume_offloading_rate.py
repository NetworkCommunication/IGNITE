"""
@File: main_consume_offloading_rate.py
@author: Zijia Zhao
@Describtion: Comparative experiments of total consumption and unloading rate
"""
from matplotlib import pyplot as plt
import pandas as pd
from env_gk_ddpg import ENV_gk_ddpg
from env_gk_dqn import ENV_gk_dqn
from env_gk_ddpg_local import ENV_gk_ddpg_local
from env_gk_ddpg_full import ENV_gk_ddpg_full
from env_nogk_ddpg import ENV_nogk_ddpg
from network import Agent
import numpy as np
from dqn import DQN
from dev.contrast import dqn

num_car = 15
num_scar = int(num_car * 1 / 3)
num_tcar = num_car - num_scar
num_task = num_tcar
num_bs = 1

def gk_ddpg(episode, filename, step):
    env = ENV_gk_ddpg(num_car, num_tcar, num_scar, num_task, num_bs)
    n_actions = 4
    n_state = num_tcar * 3 + num_car + num_bs
    MECSnet = Agent(alpha=0.0004, beta=0.004, input_dims=n_state,
                    tau=0.01, env=env, batch_size=64, layer1_size=500,
                    layer2_size=300, n_actions=n_actions)

    count_record = []
    count_record_step = []
    episode_record = []
    episode_record_step = []
    cost_record = []
    cost_record_step = []
    print('服务车辆数量： {}'.format(num_scar))
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
        print('episode ', i, 'score %.2f' % score, "    wrong: ", env.count_wrong)
        count_record.append(1 - env.count_wrong / num_task)
        if i % step == 0:
            MECSnet.save_models()
            episode_record_step.append(i)
            cost_record_step.append(np.mean(cost_record))
            count_record_step.append(np.mean(count_record))

    df_cost = pd.DataFrame({"Episode": episode_record_step, "Cost": cost_record_step}).set_index('Episode')
    df_cost.to_excel("gk_ddpg/consume/{}.xlsx".format(filename+'_consume_gk_ddpg'))

    df_count = pd.DataFrame({"Episode": episode_record_step, "Rate": count_record_step}).set_index('Episode')
    df_count.to_excel("gk_ddpg/offloading_rate/{}.xlsx".format(filename+'_offloading_rate_gk_ddpg'))

    plt.figure()
    x_data = range(len(cost_record_step))
    plt.plot(x_data, cost_record_step)

    plt.figure()
    x_data = range(len(count_record_step))
    plt.plot(x_data, count_record_step)

    plt.show()

def gk_dqn(episode, filename, step):
    env = ENV_gk_dqn(num_car, num_task, num_bs)
    deep_dqn = DQN()

    count_record = []
    count_record_step = []
    episode_record = []
    episode_record_step = []
    cost_record = []
    cost_record_step = []

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

        print('episode ', i, 'score %.2f' % score, "    wrong: ", env.count_wrong)
        count_record.append(1 - env.count_wrong / num_task)
        if i % step == 0:
            episode_record_step.append(i)
            cost_record_step.append(np.mean(cost_record))
            count_record_step.append(np.mean(count_record))

    df_cost = pd.DataFrame({"Episode": episode_record_step, "Cost": cost_record_step}).set_index('Episode')
    df_cost.to_excel("gk_dqn/consume/{}.xlsx".format(filename+'_consume_gk_dqn'))

    df_count = pd.DataFrame({"Episode": episode_record_step, "Rate": count_record_step}).set_index('Episode')
    df_count.to_excel("gk_dqn/offloading_rate/{}.xlsx".format(filename+'_offloading_rate_gk_dqn'))

    plt.figure()
    x_data = range(len(cost_record_step))
    plt.plot(x_data, cost_record_step)

    plt.figure()
    x_data = range(len(count_record_step))
    plt.plot(x_data, count_record_step)

    plt.show()

def gk_ddpg_local(episode, filename, step):
    env = ENV_gk_ddpg_local(num_car, num_tcar, num_scar, num_task, num_bs)
    n_actions = 4
    n_state = num_tcar * 3 + num_car + num_bs
    MECSnet = Agent(alpha=0.0004, beta=0.004, input_dims=n_state,
                    tau=0.01, env=env, batch_size=64, layer1_size=500,
                    layer2_size=300, n_actions=n_actions)

    count_record = []
    count_record_step = []
    episode_record = []
    episode_record_step = []
    cost_record = []
    cost_record_step = []
    print('服务车辆数量： {}'.format(num_scar))
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
        print('episode ', i, 'score %.2f' % score, "    wrong: ", env.count_wrong)
        count_record.append(1 - env.count_wrong / num_task)
        if i % step == 0:
            MECSnet.save_models()
            episode_record_step.append(i)
            cost_record_step.append(np.mean(cost_record))
            count_record_step.append(np.mean(count_record))

    df_cost = pd.DataFrame({"Episode": episode_record_step, "Cost": cost_record_step}).set_index('Episode')
    df_cost.to_excel("gk_ddpg_local/consume/{}.xlsx".format(filename+'_consume_gk_ddpg_local'))

    df_count = pd.DataFrame({"Episode": episode_record_step, "Rate": count_record_step}).set_index('Episode')
    df_count.to_excel("gk_ddpg_local/offloading_rate/{}.xlsx".format(filename+'_offloading_rate_gk_ddpg_local'))

    plt.figure()
    x_data = range(len(cost_record_step))
    plt.plot(x_data, cost_record_step)

    plt.figure()
    x_data = range(len(count_record_step))
    plt.plot(x_data, count_record_step)

    plt.show()

def gk_ddpg_full(episode, filename, step):
    env = ENV_gk_ddpg_full(num_car, num_tcar, num_scar, num_task, num_bs)
    n_actions = 4
    n_state = num_tcar * 3 + num_car + num_bs
    MECSnet = Agent(alpha=0.0004, beta=0.004, input_dims=n_state,
                    tau=0.01, env=env, batch_size=64, layer1_size=500,
                    layer2_size=300, n_actions=n_actions)

    count_record = []
    count_record_step = []
    episode_record = []
    episode_record_step = []
    cost_record = []
    cost_record_step = []
    print('服务车辆数量： {}'.format(num_scar))
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
        print('episode ', i, 'score %.2f' % score, "    wrong: ", env.count_wrong)
        count_record.append(1 - env.count_wrong / num_task)

        if i % step == 0:
            MECSnet.save_models()
            episode_record_step.append(i)
            cost_record_step.append(np.mean(cost_record))
            count_record_step.append(np.mean(count_record))

    df_cost = pd.DataFrame({"Episode": episode_record_step, "Cost": cost_record_step}).set_index('Episode')
    df_cost.to_excel("gk_ddpg_full/consume/{}.xlsx".format(filename+'_consume_gk_ddpg_full'))

    df_count = pd.DataFrame({"Episode": episode_record_step, "Rate": count_record_step}).set_index('Episode')
    df_count.to_excel("gk_ddpg_full/offloading_rate/{}.xlsx".format(filename+'_offloading_rate_gk_ddpg_full'))

    plt.figure()
    x_data = range(len(cost_record_step))
    plt.plot(x_data, cost_record_step)

    plt.figure()
    x_data = range(len(count_record_step))
    plt.plot(x_data, count_record_step)

    plt.show()

def nogk_ddpg(episode, filename, step):
    car = 25
    tcar = 10
    scar = 15
    bs = 1
    task = 10
    env = ENV_nogk_ddpg(car, tcar, scar, task, bs)
    n_actions = 4
    n_state = tcar * 3 + car + bs
    MECSnet = Agent(alpha=0.0004, beta=0.004, input_dims=n_state, tau=0.01, env=env,
                    batch_size=64, layer1_size=500, layer2_size=300, n_actions=n_actions)

    count_record = []
    count_record_step = []
    episode_record = []
    episode_record_step = []
    cost_record = []
    cost_record_step = []
    print('服务车辆数量： {}'.format(num_scar))
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
        print('episode ', i, 'score %.2f' % score, "    wrong: ", env.count_wrong)
        count_record.append(1 - env.count_wrong / num_task)
        if i % step == 0:
            MECSnet.save_models()
            episode_record_step.append(i)
            cost_record_step.append(np.mean(cost_record))
            count_record_step.append(np.mean(count_record))

    df_cost = pd.DataFrame({"Episode": episode_record_step, "Cost": cost_record_step}).set_index('Episode')
    df_cost.to_excel("nogk_ddpg/consume/{}.xlsx".format(filename+'_consume_nogk_ddpg'))

    df_count = pd.DataFrame({"Episode": episode_record_step, "Rate": count_record_step}).set_index('Episode')
    df_count.to_excel("nogk_ddpg/offloading_rate/{}.xlsx".format(filename+'_offloading_rate_nogk_ddpg'))

    plt.figure()
    x_data = range(len(cost_record_step))
    plt.plot(x_data, cost_record_step)

    plt.figure()
    x_data = range(len(count_record_step))
    plt.plot(x_data, count_record_step)
    plt.show()

if __name__ == '__main__':
    episode = 50
    step = 25
    round = 1

    filename = str(round)
    gk_ddpg(episode, filename, step)
    gk_dqn(episode, filename, step)
    gk_ddpg_local(episode, filename, step)
    gk_ddpg_full(episode, filename, step)
    nogk_ddpg(episode, filename, step)