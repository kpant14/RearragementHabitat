import time
from collections import deque

import os

from habitat.core.dataset import Episode

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List
import gym
import logging
from arguments import get_args
from env import make_vec_envs
from utils.storage import GlobalRolloutStorage, FIFOMemory
from utils.optimization import get_optimizer
from model import RL_Policy, Local_IL_Policy, Neural_SLAM_Module
from env.habitat.utils import visualizations as viztools
import algo

import sys
import matplotlib

if sys.platform == 'darwin':
    matplotlib.use("tkagg")
import matplotlib.pyplot as plt

# plt.ion()
# fig, ax = plt.subplots(1,4, figsize=(10, 2.5), facecolor="whitesmoke")


args = get_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = args.device = torch.device("cuda:0" if args.cuda else "cpu")

if args.cuda:
    torch.cuda.manual_seed(args.seed)


def main():   
    # Starting environments
    output_path = "/home/coral/Habitat_test/output/"
    num_episodes = int(args.num_episodes)
    obsList = []
    torch.set_num_threads(1)
    envs = make_vec_envs(args)
    obs, infos = envs.reset()
    
    print(envs.action_space)
    try:
        
        obsList.append(obs.numpy())
        for ep_num in range(num_episodes):
            for step in range(args.max_episode_length):
                
                #obs, infos = envs.reset()
                #print(envs.action_space.sample())
                #action  = [envs.action_space.sample()]
                action = [np.random.randint(0,4)]
                #print(action[0])
                # if (action[0] == 4):
                #     obs, infos = envs.reset()
                #     print("yes")
                #     continue
                #action = [0]
                obs, rew, done, infos = envs.step(action)
                obsList.append(obs.numpy())
        
    except KeyboardInterrupt:
        pass
    viztools.make_video_cv2(output_path,obsList, None, prefix="test", open_vid=True)
if __name__ == "__main__":
    main()