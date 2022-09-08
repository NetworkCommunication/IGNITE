"""
-*- coding: utf-8 -*-
@author: Zijia Zhao
@Describtion:Entity class and Initialization method
"""
import random

def get_car_info(num):
    md = {}
    i = 0
    while i < num:
        md[i] = Car(5e7, random.randint(1, 10), 0.5e8)
        i += 1
    return md


class Car:
    def __init__(self, cpu_frequency, service_money, fn):
        self.cpu_frequency = cpu_frequency
        self.service_money = service_money
        self.fn = fn


def get_task_info(num):
    task = {}
    i = 0
    while i < num:
        cn = random.randint(1e7, 1e8)
        task[i] = Task(i, 2*cn, cn, 1.5)
        i += 1
    return task


class Task:
    def __init__(self, id, dn, cn, delay_constraints):
        self.id = id
        self.dn = dn
        self.cn = cn
        self.delay_constraints = delay_constraints


def get_bs_info(num):
    bs = {}
    i = 0
    while i < num:
        bs[i] = BS(1e8, random.randint(2e8, 1e9))
        i += 1
    return bs


class BS:
    def __init__(self, cpu_frequency, fn_bs):
        self.cpu_frequency = cpu_frequency
        self.fn_bs = fn_bs