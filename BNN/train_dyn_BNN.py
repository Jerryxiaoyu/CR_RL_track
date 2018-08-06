
import argparse
import torch
import gym
import ast

import numpy as np

from core import utils
from core import controller
from core.model.BNN import  BNN3
from core.model.BNN_h import BNN3_h
from core.utils import log,logging_output
from core.base.ExperienceDataset import DataBuffer
from core.base.base import train_dynamics_model,learn_policy,rollout,test_episodic_cost, train_dynamics_model_pilco,train_dynamics_model_pilco2
#from MB.eval_model import plot_train_ion
#from MB.eval_model import plot_train, plot_train_std,load_data
from core.my_envs.cartpole_swingup import *

from my_envs.mujoco import *

torch.set_default_tensor_type('torch.Tensor')

parser = argparse.ArgumentParser(description='DeepPILCO')
parser.add_argument('--seed', type=int, default=1)


#ENV
parser.add_argument('--env_name', type=str, default='HalfCheetahTrack-v2')  #  Ant2-v2  HalfCheetah-v2  ArmReacherEnv-v0  Hopper2-v2: Swimmer2-v2 CartpoleSwingUp-v0
parser.add_argument('--max_timestep', type=int, default=1000)
# Dynamics
parser.add_argument('--hidden_size', type=int, default= 500)
parser.add_argument('--num_hidden_layers', type=int, default=3)
parser.add_argument('--num_itr_dyn', type=int, default=1000)  # epoch
parser.add_argument('--dyn_batch_size', type=int, default=100)  # batch_size
parser.add_argument('--lr_dynamics', type=float, default=1e-5)
parser.add_argument('--drop_p', type=float, default=0.00)
parser.add_argument('--dyn_reg2', type=float, default= 0 )
parser.add_argument('--pre_process', type=ast.literal_eval, default= True)
parser.add_argument('--net_activation', type=str, default= 'relu')

#EXP
parser.add_argument('--n_rnd', type=int, default=5) #5
parser.add_argument('--exp_num', type=str, default='1')#5
parser.add_argument('--LengthOfCurve', type=int, default=100)
parser.add_argument('--exp_group_dir', type=str, default= None )

args = parser.parse_args()
print(args)

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
exp_group_dir = args.exp_group_dir

# Exp paramaters
log_interval_policy = 100
exp_name ='BNN_modle'
exp_num =args.exp_num
log_name = 'train._{} _drop{}_nrd{}_Pre{}-EXP_{}'.format( exp_name,
                                                   drop_p, n_rnd, pre_process,exp_num)

# Create log files
log_dir = utils.configure_log_dir(env_name, txt=log_name,  No_time = False, log_group=  exp_group_dir)
logger = utils.Logger(log_dir, csvname='log_loss')

logging_output(log_dir)
# save args prameters
with open(log_dir + '/info.txt', 'wt') as f:
    print('Hello World!\n', file=f)
    print(args, file=f)
    

# Set up environment
env = gym.make(env_name)

# Create dynamics model
dynamics = BNN3(env, hidden_size=[hidden_size]*num_hidden_layers, drop_prob=drop_p, activation= net_activation) .cuda()
dynamics_optimizer =  torch.optim.Adam(dynamics.parameters(), lr= lr_dynamics, weight_decay=dyn_reg2 )



# Create random policy
randpol = controller.RandomPolicy(env)

# Create Data buffer
exp_data = DataBuffer(env, max_trajectory = 100)

# during first n_rnd trials, apply randomized controls
for i in range(n_rnd):
    exp_data.push(rollout(env, randpol, max_steps=T,render=False))

log.infov('-----------------DeepPILCO Iteration # {}-----------------' )

# Train dynamics

train_dynamics_model_pilco(dynamics, dynamics_optimizer, exp_data, epochs=num_itr_dyn, batch_size=dyn_batch_size,
                           plot_train= None, pre_process= pre_process, logger = logger)  #plot_train_ion
 
# Save model
save_dir = log_dir
utils.save_net_param(dynamics, save_dir, name='dyn_model'  , mode='net')

#
# save_dir = log_dir
# (_, _), (x_test, y_test) = load_data()
# plot_train(x_test, y_test, dyn_model=dynamics, pre_process=pre_process, save=False,
#            save_dir=save_dir + '/dyn_fig0.jpg', LengthOfCurve=LengthOfCurve)
# (_, _), (x_test, y_test) = load_data(dir_name = '/home/drl/PycharmProjects/DeployedProjects/deepPILCO/MB/data/log-test1.csv',data_num =1000)
# plot_train(x_test, y_test, dyn_model=dynamics, pre_process=pre_process, save=True,
#            save_dir=save_dir + '/dyn_fig_expect.jpg', LengthOfCurve=LengthOfCurve)
#
# plot_train_std(x_test, y_test, dyn_model=dynamics, pre_process=pre_process, save=True,
#            save_dir=save_dir + '/dyn_fig_std.jpg', LengthOfCurve=LengthOfCurve)
 
