# IGNITE

## Project Introduction

Vehicles on the road are clustered using DT techniques and Gkmeans (an optimized kmeans algorithm that does not use distance alone as the only criterion, but uses the gravitational replacement distance between vehicles as defined in this thesis) clustering algorithm to find the optimal unloading space. The DDPG reinforcement learning algorithm is used to achieve the requirement of optimal total offloading consumption by optimizing the task offloading rate and vehicle service price. A series of comparison experiments are also conducted in this project, firstly for other reinforcement learning algorithms such as DQN, secondly for the vehicle parameters, RSU parameters and unloading rates set in the experiments are changed and compared.
Finally, different reinforcement learning objectives, such as time cost and total consumption cost, are set and compared.

## Environmental Dependence

The code requires python3 (>=3.6) with the development headers. The code also need system packages as bellow:

tensflow ==2.9.1，

os，

torch == 1.11.0，

openpyxl == 3.0.10，

 matplotlib == 3.5.2，

numpy == 1.22.4，

pandas == 1.4.2，

gym == 0.18.0. 

If users encounter environmental problems and reference package version problems that prevent the program from running, please refer to the above installation package and corresponding version.



## How to Run

Each file in the dev folder is independent, for example, if you want to run the code in the ddpg file, run run.py, and if the above installation packages are successfully installed, you can run the code. DQN and DDPG files are run through **run.py**, contrast and contrast_time are executed through **Python files starting with main**.

## Catalog Structure

### contrast

With DQN algorithm combined with Gkmeans, DDPG algorithm combined with full unloading rate and Gkmeans, DDPG algorithm combined with all local unloading method and Gkmeans, DDPG algorithm without Gkmeans, etc. Different parameter settings are performed separately for comparison experiments, and the reinforcement learning objective is the total consumption cost.

**File description:**

- Files starting with env are the environmental requirements of different comparative experiments.
- Files starting with main are the initiation functions for running different comparative experiments.
- K-meas files are traditional k-means algorithm files.
- GKmeans is an improved file of k-means algorithm.

### contrast_time

Comparison experiments with DQN algorithm combined with Gkmeans, DDPG algorithm combined with full unloading rate and Gkmeans, DDPG algorithm combined with all local unloading method and Gkmeans, DDPG algorithm without Gkmeans, etc. with different parameter settings respectively, and the reinforcement learning objective is the total time cost.

**File description:**

- Files starting with env are the environmental requirements of different comparative experiments.
- Files starting with main are the initiation functions for running different comparative experiments.

### ddpg

Using ddpg algorithm + Gkemans clustering algorithm to achieve vehicle clustering and partial unloading.ddpg.

**File description:**

- The env.py in the file is the environmental configuration of the DDPG algorithm.
- The GKmeans file refers to the description in the paper.
- The network file is the network structure of the DDPG algorithm.
- The other file is the information configuration of the vehicle, task and base station. 
- The run file is the file where the algorithm is run.

### dqn

Vehicle partial unloading using dqn algorithm.

**File description:**

- The env.py in the file is the environmental configuration of the DQN algorithm.
- dqn file is a file after parameter changes to DQN.
- dqn_normal file is a DQN algorithm for reference.
- The other file is the information configuration of the vehicle, task and base station. 
- The run file is the file where the algorithm is run.

## Statement
In this project, due to the different parameter settings of vehicle, task, RSU, etc., the parameters of reinforcement learning algorithm are set differently, and the reinforcement learning process is different, resulting in different experimental results.
