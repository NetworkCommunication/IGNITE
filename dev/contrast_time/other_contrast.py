import random
def get_car_info(num, fn_car):
    md = {}
    i = 0
    while i < num:
        md[i] = Car(5e7, random.randint(1, 10), fn_car)
        i += 1
    return md

class Car:
    def __init__(self, cpu_frequency, service_money, fn):
        self.cpu_frequency = cpu_frequency
        self.service_money = service_money
        self.fn = fn

def get_task_info(num, cn):
    task = {}
    i = 0
    if cn == 0:
        while i < num:
            cn_true = random.randint(1e7, 1e8)
            task[i] = Task(i, 2 * cn_true, cn_true, 1.5)
            i += 1
    else:
        while i < num:
            task[i] = Task(i, 2*cn, cn, 1.5)
            i += 1
    return task

class Task:
    def __init__(self, id, dn, cn, delay_constraints):
        self.id = id
        self.dn = dn
        self.cn = cn
        self.delay_constraints = delay_constraints

def get_bs_info(num, fn_bs):
    bs = {}
    i = 0
    if fn_bs == 0:
        while i < num:
            bs[i] = BS(1e8, random.randint(2e8, 1e9))
            i += 1
    else:
        while i < num:
            bs[i] = BS(1e8, fn_bs)
            i += 1
    return bs

class BS:
    def __init__(self, cpu_frequency, fn_bs):
        self.cpu_frequency = cpu_frequency
        self.fn_bs = fn_bs