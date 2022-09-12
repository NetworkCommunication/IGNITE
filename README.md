# IGNITE
## Project Introduction
Vehicles on the road are clustered using DT techniques and Gkmeans (an optimized kmeans algorithm that does not use distance alone as the only criterion, but uses the gravitational replacement distance between vehicles as defined in this thesis) clustering algorithm to find the optimal unloading space. The DDPG reinforcement learning algorithm is used to achieve the requirement of optimal total offloading consumption by optimizing the task offloading rate and vehicle service price. A series of comparison experiments are also conducted in this project, firstly for other reinforcement learning algorithms such as DQN, secondly for the vehicle parameters, RSU parameters and unloading rates set in the experiments are changed and compared.
Finally, different reinforcement learning objectives, such as time cost and total consumption cost, are set and compared.
## Environmental dependence
The code requires python3 (>=3.6) with the development headers. The code also need system packages tensflow, os, torch, openpyxl, matplotlib, numpy pandas and gym. 
## Catalog Structure
### contrast
with DQN algorithm combined with Gkmeans, DDPG algorithm combined with full unloading rate and Gkmeans, DDPG algorithm combined with all local unloading method and Gkmeans, DDPG algorithm without Gkmeans, etc. Different parameter settings are performed separately for comparison experiments, and the reinforcement learning objective is the total consumption cost.
### contrast_time
comparison experiments with DQN algorithm combined with Gkmeans, DDPG algorithm combined with full unloading rate and Gkmeans, DDPG algorithm combined with all local unloading method and Gkmeans, DDPG algorithm without Gkmeans, etc. with different parameter settings respectively, and the reinforcement learning objective is the total time cost.
### ddpg
using ddpg algorithm + Gkemans clustering algorithm to achieve vehicle clustering and partial unloading
### dqn
Vehicle partial unloading using dqn algorithm
## Statement
In this project, due to the different parameter settings of vehicle, task, RSU, etc., the parameters of reinforcement learning algorithm are set differently, and the reinforcement learning process is different, resulting in different experimental results
