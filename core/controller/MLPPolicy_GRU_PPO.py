
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import torch.nn.functional as F
import math

class MLPPolicy(nn.Module):
    """MLP Policy for the controller"""
    
    def __init__(self, env, hidden_size=50):
        super().__init__()
        
        self.env = env
        
        self.hidden_size = hidden_size
        
        # Fully connected layers
        # self.fc1 = nn.Linear(in_features=env.observation_space.shape[0] + 1,  # [x, dx, polex, poley, dtheta]
        #                      out_features=self.hidden_size,
        #                      bias=True)
        # self.out = nn.Linear(in_features=self.hidden_size,
        #                      out_features=1,  # 1D continuous action space [mu, log-space of sigma]
        #                      bias=True)

        self.fc1 = nn.Linear(in_features=env.observation_space.shape[0]  ,  # State + Action
                             out_features=self.hidden_size,
                             bias=True)
        self.fc2 = nn.Linear(in_features=self.hidden_size,
                             out_features=self.hidden_size,
                             bias=True)
        self.out = nn.Linear(in_features=self.hidden_size,
                             out_features=env.action_space.shape[0],  # Next state
                             bias=True)
    
    def forward(self, x):
        # polex = torch.sin(x[:, 2]) * 0.6
        # poley = torch.cos(x[:, 2]) * 0.6
        #
        # x = torch.stack([x[:, 0], x[:, 1], x[:, 0] + polex, poley, x[:, 3]], 1)
        #
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        # mu = x[:, 0]
        # std = F.sigmoid(x[:, 1])  # Softplus vs sigmoid
        # z = Variable(torch.randn(mu.size()))

        # action = mu + std*z ###### For now, try deterministic actions
        
        # x = 9 / 8 * torch.sin(x) + 1 / 8 * torch.sin(3 * x)
        # x = torch.tanh(x)
        x = x * self.env.action_space.high[0]
        x = torch.clamp(x[:, 0], self.env.action_space.low[0], self.env.action_space.high[0])
        
        return x # x
def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


class BNNPolicyGRU_PPO(nn.Module):
    """
    Learning dynamics model via regression

    fro complex module

    """
    def __init__(self, env, hidden_size=[64] * 3, drop_prob=0.0, log_std= 0):
        super().__init__()
        
        self.env = env
        # Flag for sampling parameters
        self.sampling = False
        # Fix the random mask for dropout, each batch contains K particles
        self.mask = []
        self.hidden_size = hidden_size
        
        for i in range(len(hidden_size)):
            self.mask.append(None)
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.input_dim = self.obs_dim
        self.ouput_dim =  self.act_dim
        
        self.drop_prob = drop_prob

        self.sequence_length =1
        self.batch_size =1
        #self.rnn = nn.GRU(input_size=self.input_dim, hidden_size=hidden_size[0], num_layers= len(hidden_size), batch_first=True)
        self.dropout = torch.nn.Dropout(p=self.drop_prob)
        
        
        self.fc_layers = torch.nn.ModuleList()
        last_dim = self.input_dim
        for nh in hidden_size:
            self.fc_layers.append(nn.GRU(input_size=last_dim, hidden_size=nh,  batch_first=True))
            last_dim = nh

        self._set_init_param()
        self.action_mean = nn.Linear(last_dim, self.ouput_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.ones(1, self.ouput_dim ) * log_std)
        
        
    
    def forward(self, x, h0=None,  training=True ):
        batch_size = x.shape[0]
        x = x.view( batch_size, self.sequence_length, self.input_dim)
  
        if self.sampling:
            for i, affine in enumerate(self.fc_layers):
                # Check if drop mask with correct dimension
                if self.mask[i].size()[0] != x.size()[0]:
                    raise ValueError('Dimension of fixed masks must match the batch size.')
                x, hidden =  affine(x)
                x = x * self.mask[i]
        else:
            for affine in self.fc_layers:
                x, hidden = affine(x)
                x  = self.dropout(x)
         

        action_mean = self.action_mean(x).view((batch_size,-1))
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
 
        return action_mean, action_log_std, action_std

    def select_action(self, x):
        action_mean, _, action_std = self.predict_Y(x)
        
        action = torch.normal(action_mean, action_std)
    
        return action.data

    def get_log_prob(self, x, actions):
        action_mean, action_log_std, action_std = self.forward(x)
        return normal_log_density(actions, action_mean, action_log_std, action_std)
    
    def set_sampling(self, sampling=None, batch_size=None):
        if sampling is None:
            raise ValueError('Sampling cannot be None.')
        
        self.sampling = sampling
        
        if self.sampling:
            # Sample dropout random masks
            for i, affine in enumerate(self.fc_layers):
                self.mask[i] = Variable(torch.bernoulli(torch.zeros(batch_size,self.sequence_length, self.hidden_size[i]).fill_(1 - self.drop_prob))).cuda()
                # Rescale by 1/p to maintain output magnitude
                self.mask[i] /= (1 - self.drop_prob)
    
    def _set_init_param(self):
        for param in self.parameters():
            nn.init.normal(param, mean=0, std=1e-2)
    
    def predict_Y(self, x,  pre_prcess=True):
        if pre_prcess:
            x = (x - self.Xm) / self.Xstd
        # Forward pass
        action_mean, action_log_std, action_std = self.forward(x )
        return action_mean, action_log_std, action_std
 
    
    def update_dataset_statistics(self, exp_dataset):
        state = exp_dataset.data[:, :exp_dataset.observation_dim]
        self.Xm = np.atleast_1d(state.mean(0))
        self.Xstd = np.atleast_1d(state.std(0))

        # from numpy to torch
        self.Xm = Variable(torch.from_numpy(self.Xm).float().cuda())
        self.Xstd = Variable(torch.from_numpy(self.Xstd).float().cuda())
 
   