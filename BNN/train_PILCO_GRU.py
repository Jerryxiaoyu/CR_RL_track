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

parser = argparse.ArgumentParser(description='DeepPILCO')
parser.add_argument('--seed', type=int, default=1)

#ENV
parser.add_argument('--env_name', type=str, default='HalfCheetahTrack-v2')  #  Ant2-v2  HalfCheetah-v2  ArmReacherEnv-v0  Hopper2-v2: Swimmer2-v2
parser.add_argument('--max_timestep', type=int, default=1000)
# Dynamics
parser.add_argument('--hidden_size', type=int, default=200 )
parser.add_argument('--num_hidden_layers', type=int, default=2)
parser.add_argument('--num_itr_dyn', type=int, default=50)  # epoch
parser.add_argument('--dyn_batch_size', type=int, default=1024)  # batch_size
parser.add_argument('--lr_dynamics', type=float, default=1e-3)
parser.add_argument('--drop_p', type=float, default=0.1)
parser.add_argument('--dyn_reg2', type=float, default=0.00001 )
parser.add_argument('--pre_process', type=ast.literal_eval, default= True)
parser.add_argument('--net_activation', type=str, default= 'relu')

#EXP
parser.add_argument('--n_rnd', type=int, default=10) #5
parser.add_argument('--exp_num', type=str, default='1')#5
parser.add_argument('--LengthOfCurve', type=int, default=100)
parser.add_argument('--num_iter_algo', type=int, default=10)
parser.add_argument('--num_iter_policy', type=int, default=50 )
# Policy
parser.add_argument('--lr_policy', type=float, default=1e-3)
parser.add_argument('--policy_type', type=str, default='ActorModel')


args = parser.parse_args()
print(args)

shaping_state_delta = False


env_name = args.env_name

T = args.max_timestep

hidden_size = args.hidden_size
drop_p = args.drop_p
lr_dynamics = args.lr_dynamics
n_rnd =args.n_rnd
num_hidden_layers =args.num_hidden_layers
num_itr_dyn =args.num_itr_dyn
dyn_batch_size =args.dyn_batch_size
dyn_reg2= args.dyn_reg2
net_activation = args.net_activation
pre_process = args.pre_process
LengthOfCurve = args.LengthOfCurve

lr_policy = args.lr_policy


log_interval_policy = 10
exp_name ='PILCO'
num_exp =1
log_name = 'train._{}_lrp{}_drop{}-EXP_{}_GRU'.format( exp_name,args.lr_policy,
                                                     args.drop_p, num_exp )
num_iter_algo =args.num_iter_algo

num_iter_policy = args.num_iter_policy
grad_clip = 1

K = 20

# Create log files
log_dir = utils.configure_log_dir(env_name, txt=log_name,  No_time = False)
logger = utils.Logger(log_dir, csvname='log_loss')

logging_output(log_dir)
# save args prameters
with open(log_dir + '/info.txt', 'wt') as f:
    print('Hello World!\n', file=f)
    print(args, file=f)
 

# Set up environment
env = gym.make(env_name)

# Create dynamics model
dynamics = BNN3(env, hidden_size=[hidden_size] *num_hidden_layers, drop_prob=drop_p, activation= net_activation, shaping_state_delta = shaping_state_delta).cuda()
dynamics_optimizer =  torch.optim.Adam(dynamics.parameters(), lr= lr_dynamics, weight_decay=dyn_reg2 )

# Create random policy
randpol = controller.RandomPolicy(env)

# Create Policy
policy = controller.BNNPolicyGRU(env, hidden_size=[64,64,64], drop_prob=0.1, activation= 'relu').cuda()
policy_optimizer = optim.Adam(policy.parameters(), lr=lr_policy, weight_decay =1e-5 )  # 1e-2, RMSprop

# initiation
for name, param in policy.named_parameters():
    #print(name)
    if name.find('weight') != -1:
        # weight init
        nn.init.orthogonal(param)
        #nn.init.normal(param, mean=0, std=1e-2)
        #print(param)
    elif name.find('bias') != -1:
        # bias init
        nn.init.normal(param, mean=0, std=1e-2)
        #print(param)
    else:
        print('Init error')


# Create Data buffer
exp_data = DataBuffer(env, max_trajectory = num_iter_algo*10 +n_rnd,   shaping_state_delta = shaping_state_delta)


# during first n_rnd trials, apply randomized controls
for i in range(n_rnd):
    exp_data.push(rollout(env, randpol, max_steps=T))

#cost_mean ,cost_std = test_episodic_cost(env, policy, N=50, T=T, render=False)


for i in range( num_iter_algo):
    log.infov('-----------------DeepPILCO Iteration # {}-----------------'.format(i+1))
    # Train dynamics
    train_dynamics_model_pilco(dynamics, dynamics_optimizer, exp_data, epochs=num_itr_dyn, batch_size=dyn_batch_size, plot_train=None, pre_process=pre_process)

    # Update policy
    log.infov('Policy optimization...' )
    
    for j in range(num_iter_policy):
        _, list_costs, list_moments = learn_policy_pilco(env, dynamics, policy, policy_optimizer, K=K, T= 100, gamma=0.99,
                                                   moment_matching=True,   grad_norm = grad_clip, pre_prcess=True , shaping_state_delta= shaping_state_delta)

        # Loggings
        if (j + 1) % log_interval_policy == 1 or (j + 1) == args.num_iter_policy:

            loss_mean = torch.sum( torch.cat(list_costs)) .data.cpu().numpy()[0]
            grad_norm = _grad_norm(policy)
            log_str ='[Itr #{}/{} policy optim # {}/{} ]: loss mean: {:.5f},   grad norm:{:.3f}'

            log.info(log_str.format( (i+1),args.num_iter_algo,
                                  (j+1),args.num_iter_policy,
                                  loss_mean,   grad_norm ))

    cost_mean ,cost_std = test_episodic_cost2(env, policy,dynamics, N=5, T=T, render=True)
    log.info('Policy Test : # {}  cost mean {:.5f}  cost std {:.5f} '.format((i+1) ,cost_mean,cost_std ))

    # Execute system and record data
    for num in range(10):
        exp_data.push(rollout(env, policy, max_steps=T))
    
    # Save model
    save_dir = log_dir
    utils.save_net_param(policy, save_dir, name='policy_'+str(i))
    utils.save_net_param(dynamics, save_dir, name='dynamics_' + str(i))

    # Record data
    # list_ep_costs.append(torch.cat(list_costs).mean().data.cpu().numpy()[0])
    # np.savetxt(log_dir + '/ep_costs', list_ep_costs)
    # list_test_rewards.append(test_episodic_cost(env, policy, N=50, T=T, render=False))
    # np.savetxt(log_dir + '/test_rewards', list_test_rewards)
    # list_policy_param.append(next(policy.parameters()).data.cpu().numpy()[0])
    # np.savetxt(log_dir + '/policy_param', list_policy_param)
    # list_policy_grad.append(next(policy.parameters()).grad.data.cpu().numpy()[0])
    # np.savetxt(log_dir + '/policy_grad', list_policy_grad)

    logger.log({'itr': i,
                'policy_loss': torch.cat(list_costs).mean().data.cpu().numpy()[0],
                'cost_mean': cost_mean,
                'cost_std':cost_std,
                'policy_param':next(policy.parameters()).data.cpu().numpy()[0],
                'policy_grad':next(policy.parameters()).grad.data.cpu().numpy()[0]
                })
    logger.write(display=False)

logger.close()






