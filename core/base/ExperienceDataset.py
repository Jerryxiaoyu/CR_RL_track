import torch
import torch.utils.data as data
from torch.autograd import Variable

import numpy as np
from collections import namedtuple
import random
Transition = namedtuple('Transition', ('state', 'action',   'next_state',
                                       'reward','mask'))

class DataBuffer(data.Dataset):
    def __init__(self, env, max_trajectory= None, shaping_state_delta = False):
        self.data = None
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.data_set = []

        self.max_trajectory = max_trajectory  # Same as DeepPILCO
        self.buffer = []
        self.shaping_state_delta = shaping_state_delta
        if self.shaping_state_delta:
            self.shaping_state_constant = 9
        else:
            self.shaping_state_constant = 3

        self.memory = []

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        """Output FloatTensor"""
        data =  self.data[index]
        # Convert to FloatTensor
        data = torch.Tensor(data)
        
        state = data[:self.observation_dim]
        target = data[self.observation_dim + self.action_dim:self.observation_dim*2+self.action_dim]
        
        
        # return target data as difference between current and predicted state
        return data[:self.observation_dim+self.action_dim], target[:self.observation_dim-self.shaping_state_constant] - state[:self.observation_dim-self.shaping_state_constant]
    
    def push(self, D ):
        self.data_set.append(D)

        # self.memory_push(D[:,:self.observation_dim], D[:,self.observation_dim: self.observation_dim+self.action_dim],
        #                  D[:,self.observation_dim +self.action_dim :self.observation_dim*2+self.action_dim], D[:,-2],D[:,-1])

        self.buffer.append(D)
        if self.max_trajectory is not None:
            if len(self.buffer) > self.max_trajectory:
                del self.buffer[0]  # Delete oldest trajectory

        self.data = np.concatenate(self.buffer, axis=0)
        np.random.shuffle(self.data)
    
    def memory_push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))
    def sample(self, batch_size=None):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))
   
 







