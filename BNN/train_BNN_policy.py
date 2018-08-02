
import argparse
import torch
import gym

import numpy as np

from core import utils
from core import controller
from core.model.BNN import  BNN3
from core.utils import log,logging_output
from core.base.ExperienceDataset import DataBuffer
from core.base.base import train_dynamics_model,learn_policy,rollout,test_episodic_cost,train_dynamics_model_pilco
from MB.eval_model import plot_train, plot_train_std,load_data
import ast

#from my_envs.mujoco import *

#torch.set_default_tensor_type('torch.Tensor')

parser = argparse.ArgumentParser(description='DeepMPC')
parser.add_argument('--seed', type=int, default=1)


#ENVc
parser.add_argument('--env_name', type=str, default='HalfCheetah-v2')  #  Ant2-v2  HalfCheetah-v2  ArmReacherEnv-v0  Hopper2-v2: Swimmer2-v2
parser.add_argument('--max_timestep', type=int, default=1000)
# Dynamics
parser.add_argument('--hidden_size', type=int, default=1000)
parser.add_argument('--num_hidden_layers', type=int, default=2)
parser.add_argument('--num_itr_dyn', type=int, default=1000)  # epoch
parser.add_argument('--dyn_batch_size', type=int, default=1000)  # batch_size
parser.add_argument('--lr_dynamics', type=float, default=1e-3)
parser.add_argument('--drop_p', type=float, default=0.1)
parser.add_argument('--dyn_reg2', type=float, default=0.0 )
parser.add_argument('--pre_process', type=ast.literal_eval, default= True)
parser.add_argument('--net_activation', type=str, default= 'relu')

# MPC
parser.add_argument('--mpc_horizon', '-m', type=int, default=10)  # mpc simulation H  10
parser.add_argument('--simulated_paths', '-sp', type=int, default=1000)  # mpc  candidate  K 10000
#EXP
parser.add_argument('--n_rnd', type=int, default=5) #5
parser.add_argument('--n_iter_algo', type=int, default =30)
parser.add_argument('--N_MPC', type=int, default =5)#5
parser.add_argument('--exp_num', type=str, default='1')# info of exp
parser.add_argument('--LengthOfCurve', type=int, default=100)
parser.add_argument('--mode', type=str, default='random')
parser.add_argument('--action_noise', type=float, default=0.1, help= '0 -100 %')
parser.add_argument('--exp_group_dir', type=str, default= None )
args = parser.parse_args()
print(args)

#env
env_name = args.env_name
max_timestep = args.max_timestep
#dynamics
hidden_size = args.hidden_size
num_hidden_layers =args.num_hidden_layers
drop_p = args.drop_p
lr_dynamics = args.lr_dynamics
dyn_batch_size =args.dyn_batch_size
num_itr_dyn = args.num_itr_dyn
#mpc
mpc_horizon = args.mpc_horizon
simulated_paths = args.simulated_paths
#Exp
n_rnd = args.n_rnd
n_iter_algo = args.n_iter_algo
N_MPC = args.N_MPC

dyn_reg2= args.dyn_reg2
net_activation = args.net_activation

pre_process = args.pre_process

LengthOfCurve = args.LengthOfCurve
action_noise = args.action_noise
exp_group_dir = args.exp_group_dir
#TODO important
USE_PROB_PREDICT = args.mode # 'random'#'prob'

# param check
if dyn_batch_size > max_timestep:
    assert log.error('Hyper param error: dyn_batch_size must not be more than max_timestep.')


# Exp paramaters
log_interval_policy = 100
exp_name ='Train_policy'
num_exp = args.exp_num
log_name = 'train._{} _drop{}_nrd{}-EXP_{}'.format( exp_name,
                                                   drop_p, n_rnd,num_exp)

# Create log files
log_dir = utils.configure_log_dir(env_name, txt=log_name,  No_time = False, log_group=  exp_group_dir)
logger = utils.Logger(log_dir, csvname='log')
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


from MB.cost_functions import cheetah_cost_fn
from MB.controllers import MPC_rollout,MPCcontroller
cost_fn = cheetah_cost_fn
mpc_controller = MPCcontroller(env=env,
								   dyn_model=dynamics,
								   horizon=mpc_horizon,
								   cost_fn=cost_fn,
								   num_simulated_paths=simulated_paths,
                                   action_noise= action_noise,
                                    N_SAMPLES=10,
								   )

# Create Data buffer
exp_data = DataBuffer(env, max_trajectory = n_rnd + 3* N_MPC) # num_iter_algo +n_rnd  n_rnd + n_iter_algo* N_MPC

# during first n_rnd trials, apply randomized controls
for i in range(n_rnd):
    exp_data.push(rollout(env, randpol, max_steps= max_timestep))

 
log.infov('-----------------DeepPILCO Iteration # {}-----------------'.format(i + 1))
#Train dynamics
train_dynamics_model_pilco(dynamics, dynamics_optimizer, exp_data, epochs=num_itr_dyn, batch_size=dyn_batch_size,
                           plot_train= None, pre_process= pre_process )
#dynamics.update_dataset_statistics(exp_data)
# Save model
save_dir = log_dir
utils.save_net_param(dynamics, save_dir, name='dyn_model0', mode='net')

# exp_logger = utils.Logger(log_dir, csvname='exp'  )
# data = np.concatenate((exp_data.buffer[0], exp_data.buffer[1],exp_data.buffer[2],exp_data.buffer[3],exp_data.buffer[4]), axis=0)
# exp_logger.log_table2csv(data)

for itr in range(n_iter_algo):
    reward_sums = []
    for n_mpc in range (N_MPC):
        data_MPC, reward_sum = MPC_rollout(env, mpc_controller,dynamics, horizon= max_timestep, render=False, use_prob=USE_PROB_PREDICT)
        exp_data.push(data_MPC)
        
        log.info('itr {} : The num of sampling rollout : {} Accumulated Reward :{:.4f} '.format(itr, n_mpc, reward_sum))

        
        reward_sums.append(reward_sum)
    log.infov( "Itr {}/{} Accumulated Reward: {:.4f}   ".format(itr, n_iter_algo, sum(reward_sums) / N_MPC))
    # Train dynamics
    train_dynamics_model_pilco(dynamics, dynamics_optimizer, exp_data, epochs=num_itr_dyn, batch_size=dyn_batch_size,
                               plot_train=None, pre_process=pre_process ) # plot_train_ion
    # Save model
    save_dir = log_dir
    utils.save_net_param(dynamics, save_dir, name='dyn_model'+str(itr) , mode='net')
    
    # Evaluate model
    # if env_name == 'HalfCheetah-v2':
    #     (_, _), (x_test, y_test) = load_data()
    #     plot_train(x_test, y_test, dyn_model=dynamics, pre_process=pre_process, save=True, save_dir= save_dir+'/dyn_fig{}.jpg'.format(itr),LengthOfCurve=LengthOfCurve)

    
    logger.log({'itr': itr,})
    for i in range(N_MPC):
        logger.log({
                    'Accumulated_Reward{}'.format(i): reward_sums[i],
                    })
    logger.write(display=False)
logger.close()