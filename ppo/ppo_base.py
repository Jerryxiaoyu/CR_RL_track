import torch
import numpy as np
from torch.autograd import Variable
from ppo.core.common import estimate_advantages
from ppo.core.ppo import ppo_step
import math

optim_epochs = 5
optim_batch_size = 4096

def update_PPO_params(batch, i_iter, value_net, policy_net, optimizer_value,optimizer_policy,  args):
    states = torch.from_numpy(np.stack(batch.state)).float()
    actions = torch.from_numpy(np.stack(batch.action)).float()
    rewards = torch.from_numpy(np.stack(batch.reward)).float()
    masks = torch.from_numpy(np.stack(batch.mask).astype(np.float64)).float()

    use_gpu = True
    
    states, actions, rewards, masks = states.cuda(), actions.cuda(), rewards.cuda(), masks.cuda()
    
    values = value_net(Variable(states, volatile=True)).data
    fixed_log_probs = policy_net.get_log_prob(Variable(states, volatile=True), Variable(actions)).data
    
    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values,  args['gamma'],  args['tau'], use_gpu)
    
    lr_mult = max(1.0 - float(i_iter) / args['max_iter_num'], 0)
    
    """perform mini-batch PPO update"""
    optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = torch.LongTensor(perm).cuda() if use_gpu else torch.LongTensor(perm)
        
        states, actions, returns, advantages, fixed_log_probs = \
            states[perm], actions[perm], returns[perm], advantages[perm], fixed_log_probs[perm]
        
        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
            states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]
            
            ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 5, states_b, actions_b, returns_b,
                     advantages_b, fixed_log_probs_b, lr_mult,  args['learning_rate'],  args['clip_epsilon'],  args['l2_reg'])
