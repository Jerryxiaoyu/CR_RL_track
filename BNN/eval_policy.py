import argparse
import ast
import gym
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import os

from core.model.BNN import  BNN,BNN3
from core import controller
from core.base.ExperienceDataset import DataBuffer
from core.base.base import train_dynamics_model,learn_policy,rollout,test_episodic_cost, train_dynamics_model_pilco, learn_policy_pilco,test_episodic_cost2

from core.utils.utils import _grad_norm
from core.my_envs.cartpole_swingup import *
from core import utils
from core.utils import log,logging_output

from my_envs.mujoco import *


torch.set_default_tensor_type('torch.Tensor')

env_name = 'HalfCheetahTrack-v2'
T =1000


# Set up environment
env = gym.make(env_name)

# Create Policy
policy = controller.BNNPolicyGRU(env, hidden_size=[64, 200,64], drop_prob=0.1, activation= 'relu') .cuda()
#policy_optimizer = optim.Adam(policy.parameters(), lr=lr_policy, weight_decay =1e-5 )  # 1e-2, RMSprop
policy.load_state_dict(torch.load('log-files/HalfCheetahTrack-v2/Aug-06_10:49:26train._PILCO_lrp0.001_drop0.1-EXP_1_GRU/policy_0.pkl'))


cost_mean ,cost_std = test_episodic_cost2(env, policy,  N=5, T=T, render=True)
log.info('Policy Test :  cost mean {:.5f}  cost std {:.5f} '.format( cost_mean,cost_std ))