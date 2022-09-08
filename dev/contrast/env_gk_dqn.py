import numpy as np
from other import *
class ENV_gk_dqn:
    def __init__(self, num_car, num_task, num_bs):
        self.progress = None
        self.num_car = num_car
        self.num_bs = num_bs
        self.car = get_car_info(num_car)
        self.task = get_task_info(num_task)
        self.bs = get_bs_info(num_bs)
        self.num_tcar = 10
        self.num_scar = 5
        self.num_task = self.num_tcar

        self.dn = [0] * self.num_task
        self.cn = [0] * self.num_task
        self.fn = [0] * self.num_tcar
        self.fm = [0] * self.num_scar
        self.cpu_remain = [0] * (self.num_car + self.num_bs)
        self.count_wrong = 0
        self.done = False
        self.t_local = 0
        self.t_up = 0
        self.t_comp = 0
        self.t_off = 0
        self.i_task = 0
        self.reward = 0

        self.W = 10
        self.N0 = -174
        self.hn = random.uniform(1e-8, 1e-7)
        self.P = 0.1
        self.car_cpu_frequency = 5e7
        self.bs_cpu_frequency = 1e8

    def get_init_state(self):
        self.count_wrong = 0
        self.car = get_car_info(self.num_car)
        self.task = get_task_info(self.num_task)
        self.bs = get_bs_info(self.num_bs)
        self.dn = [0] * self.num_task
        self.cn = [0] * self.num_task
        if self.done:
            self.i_task = 0
        i = 0
        while i < (self.num_car + self.num_bs):
            if i == self.num_car:
                self.cpu_remain[i] = self.bs_cpu_frequency
                break
            self.cpu_remain[i] = self.car_cpu_frequency
            i += 1

        self.progress = [0] * self.num_task
        state = np.concatenate((self.dn, self.cn, self.progress, self.cpu_remain))
        return state

    def step(self, action):
        get1 = 1 if 0 <= action <= 100 else 0
        get2 = 0
        get3 = 0
        get4 = 1
        if get1 == 1:
            get2 = int(action * 6 / 100) +1
            if get2 == 0:
                get2 = 1
            get3 = int(action / 10)
            get4 = 1 + action * 0.005

        i_task = self.i_task
        T = self.task[i_task].delay_constraints
        Cpu_task = self.task[i_task].cn
        if get1 == 0 or get3 == 0:
            t = Cpu_task/self.car[i_task].fn
            if t <= T and Cpu_task <= self.car[i_task].cpu_frequency:
                self.dn[i_task] = self.task[i_task].dn
                self.cn[i_task] = self.task[i_task].cn
                self.progress[i_task] = 1
                self.done = True if sum(self.progress) == self.num_task else False
                self.cpu_remain[i_task] = self.car[i_task].cpu_frequency - Cpu_task
                self.car[i_task].cpu_frequency -= Cpu_task
                self.reward = -7 * t
            else:
                self.dn[i_task] = self.task[i_task].dn
                self.cn[i_task] = self.task[i_task].cn
                self.progress[i_task] = 1
                self.done = True if sum(self.progress) == self.num_task else False
                self.count_wrong += 1
                self.reward = -10 * T

        else:
            rate = get3 * 0.1
            cpu_local = Cpu_task * (1-rate)

            self.t_local = cpu_local / self.car[i_task].fn
            d = random.randint(500, 5000)
            rn = 1e8
            self.t_up = rate * self.task[i_task].dn / rn
            is_bs = False
            if get2 >= 6:
                f_general = self.bs[0].fn_bs
                is_bs = True
            else:
                f_general = self.car[9 + get2].fn
            self.t_comp = rate * self.task[i_task].cn / f_general
            self.t_off = max(self.t_comp + self.t_up, self.t_local)

            if is_bs:
                cpu_general = self.bs[0].cpu_frequency
                cpu_out = rate * self.bs[0].cpu_frequency
            else:
                cpu_general = self.car[self.num_tcar - 1 + get2].cpu_frequency
                cpu_out = rate * self.car[self.num_tcar - 1 + get2].cpu_frequency

            if cpu_local <= self.car[i_task].cpu_frequency and self.t_off <= T and cpu_out <= cpu_general:
                self.progress[i_task] = 1
                self.done = True if sum(self.progress) == self.num_task else False
                self.reward = -(self.t_off + 0*rate * self.car[self.num_scar-1+get2].service_money * self.task[i_task].cn+rate*get4)
                self.dn[i_task] = self.task[i_task].dn
                self.cn[i_task] = self.task[i_task].cn

                self.cpu_remain[i_task] = self.car[i_task].cpu_frequency - Cpu_task

                if is_bs:
                    self.cpu_remain[-1] = self.bs[0].cpu_frequency - Cpu_task * rate

                else:
                    self.cpu_remain[self.num_tcar-1+get2] = \
                        self.car[self.num_tcar-1+get2].cpu_frequency - Cpu_task * rate
            else:
                self.progress[i_task] = 1
                self.done = True if sum(self.progress) == self.num_task else False
                self.reward = -8*T
                self.dn[i_task] = self.task[i_task].dn
                self.cn[i_task] = self.task[i_task].cn
                self.count_wrong += 1
        self.i_task += 1
        state = np.concatenate((self.dn, self.cn, self.progress, self.cpu_remain))
        return state, self.reward, self.done

