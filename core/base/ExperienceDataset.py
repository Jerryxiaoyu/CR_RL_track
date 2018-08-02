import torch
import torch.utils.data as data
from torch.autograd import Variable

import numpy as np




class DataBuffer(data.Dataset):
    def __init__(self, env, max_trajectory= None):
        self.data = None
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.data_set = []

        self.max_trajectory = max_trajectory  # Same as DeepPILCO
        self.buffer = []
        
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
        return data[:self.observation_dim+self.action_dim], target - state
    
    def push(self, D ):
        self.data_set.append(D)

        self.buffer.append(D)
        if self.max_trajectory is not None:
            if len(self.buffer) > self.max_trajectory:
                del self.buffer[0]  # Delete oldest trajectory

        self.data = np.concatenate(self.buffer, axis=0)
        np.random.shuffle(self.data)
    
   
 







