"""
-*- coding: utf-8 -*-
@author: Zijia Zhao
@Describtion:GKmeans to group vehicles
"""
import random
import numpy as np
def get_car_info(num):
    md = {}
    i = 0
    while i < num:
        md[i] = Car(5e7, random.randint(1, 10), 0.5e8,
                    random.randint(1e7, 1e8), random.choice((-1, 1)), random.randint(0, 6))
        i += 1
    return md

class Car:
    def __init__(self, cpu_frequency, service_money, fn, cn, direction, v):
        self.cpu_frequency = cpu_frequency
        self.service_money = service_money
        self.fn = fn
        self.cn = cn
        self.direction = direction
        self.v = v

def list_To_matrix(list, col):
    j = 0
    mid = []
    data = []
    count = 0
    for i in list:
        count += 1
        if j != col:
            mid.append(i)
            j += 1
            if count==len(list):
                data.append(mid)
        elif j == col:
            data.append(mid)
            mid = []
            mid.append(i)
            j = 1
    return data

def init(num_all):
    o1 = 1
    o2 = 1
    o3 = 1
    num_car = num_all
    F_temp = []
    car = get_car_info(num_car)
    for i in range(num_car):
        for j in range(num_car):
            m_i = car.get(i).cn/car.get(i).fn
            m_j = car.get(j).cn/car.get(j).fn
            m = max(m_i/m_j, m_j/m_i)
            if car.get(i).direction == car.get(j).direction:
                w_d = 1
            else:
                w_d = 0
            if car.get(i).v == 0 or car.get(j).v == 0:
                w_v = 0
            elif car.get(i).v > car.get(j).v :
                w_v = car.get(j).v/car.get(i).v
            elif car.get(i).v < car.get(j).v:
                w_v = car.get(i).v / car.get(j).v
            else:
                w_v = 1
            w = 2/3 * w_d + 1/3 * w_v
            r = 1
            if w == 0:
                f = o1 * m / pow((o3/r), 2)
            else:
                f = o1 * m / pow((o2/w + o3/r), 2)
            if i == j:
                f = 0
            F_temp.append(f)
    F = list_To_matrix(F_temp, num_car)
    clalist = np.array(F)
    return clalist, car

def createDataSet(num_all):
    dataset = []
    temp = []
    for i in range(num_all):
        temp.append(random.randint(0, 11))
        temp.append(random.randint(0, 11))
        if temp in dataset:
            temp = []
            temp.append(random.randint(0, 11))
            temp.append(random.randint(0, 11))
        dataset.append(temp)
        temp = []
    return dataset

def getCentroidsKey(centroids, dataSet):
    count = -1
    keys = []
    for data in dataSet:
        count += 1
        if data in centroids:
            keys.append(count)
    return keys

def calcF(keys_id, num_all):
    F_list, cars = init(num_all)
    temp_f_x = []
    clalist = []
    for f_x in F_list:
        for i in range(len(f_x)):
            if i in keys_id:
                temp_f_x.append(f_x[i])
        clalist.append(temp_f_x)
        temp_f_x = []
    clalist = np.array(clalist)
    return clalist, cars

def Gkmeans_classify(dataSet, k, num_all):
    centroids = random.sample(dataSet, k)
    keys_id = getCentroidsKey(centroids,dataSet)
    clalist, cars = calcF(keys_id, num_all)
    maxFIndices = np.argmax(clalist, axis=1)
    cluster = []
    for i in range(k):
        cluster.append([])
    for i, j in enumerate(maxFIndices):
        cluster[j].append(dataSet[i])

    return centroids, cluster, cars

def Gkmeans():
    num_all = 25
    dataset = createDataSet(num_all)
    centroids, cluster,cars = Gkmeans_classify(dataset, 2, num_all)
    length = 0
    f_data = []
    for data in cluster:
        if len(data) >= length:
            f_data = data
            length = len(data)
    keys_car = getCentroidsKey(f_data, dataset)
    new_cars = {}
    j = 0
    for i in range(len(cars)):
        if i in keys_car:
            new_cars[j] = cars.get(i)
            j += 1
    return new_cars
