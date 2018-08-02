
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import torch.nn.functional as F


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


class BNNPolicy(nn.Module):
    """
    Learning dynamics model via regression

    fro complex module

    """
    
    def __init__(self, env, hidden_size=[200] * 3, drop_prob=0.0, activation='relu'):
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
        
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = F.tanh
        else:
            print('error activation!!')
        
        self.dropout = torch.nn.Dropout(p=self.drop_prob)
        self.fc_layers = torch.nn.ModuleList()
        last_dim = self.input_dim
        for nh in hidden_size:
            self.fc_layers.append(torch.nn.Linear(last_dim, nh))
            last_dim = nh
        
        self.out_layer = torch.nn.Linear(last_dim, self.ouput_dim)
        
        self._set_init_param()
    
    def forward(self, x, training=True ):
        
        
        if self.sampling:
            
            for i, affine in enumerate(self.fc_layers):
                # Check if drop mask with correct dimension
                if self.mask[i].size()[0] != x.size()[0]:
                    raise ValueError('Dimension of fixed masks must match the batch size.')
                x = self.activation(affine(x))
                x = x * self.mask[i]
        else:
            for affine in self.fc_layers:
                x = self.activation(affine(x))
                x = self.dropout(x)
        
        x = self.out_layer(x)
        
        
        
        return x
    
    def set_sampling(self, sampling=None, batch_size=None):
        if sampling is None:
            raise ValueError('Sampling cannot be None.')
        
        self.sampling = sampling
        
        if self.sampling:
            # Sample dropout random masks
            
            for i, affine in enumerate(self.fc_layers):
                self.mask[i] = Variable(
                    torch.bernoulli(
                        torch.zeros(batch_size, self.hidden_size[i]).fill_(1 - self.drop_prob))).cuda()
                # Rescale by 1/p to maintain output magnitude
                self.mask[i] /= (1 - self.drop_prob)
    
    def _set_init_param(self):
        for param in self.parameters():
            nn.init.normal(param, mean=0, std=1e-2)
    
    def predict_Y(self, x,  pre_prcess=True):
        
        if pre_prcess:
            # standardize inputs
            # x = torch.matmul(x - self.Xm, self.iXs)
            x = (x - self.Xm) / self.Xstd
        
        # Forward pass
        y = self.forward(x )
        
        if pre_prcess:
            # scale and center outputs
            # y= torch.matmul(y, self.Ys) +self.Ym
            y = y * self.Ystd + self.Ym
        
        
        
        return y
   
    
    def predict_samples(self, x,  pre_prcess=True):
        
        
        if pre_prcess:
            # standardize inputs
            x = (x - self.Xm) / self.Xstd
        
        # Forward pass
        y = self.forward(x )
        
        if pre_prcess:
            # scale and center outputs
            y = y * self.Ystd + self.Ym
         
    
        
        return y
    
    def update_dataset_statistics(self, exp_dataset):
        
        X_dataset = exp_dataset.data[:, :exp_dataset.observation_dim + exp_dataset.action_dim]
        X_dataset += 1e-6 * np.random.randn(*X_dataset.shape)
        state = X_dataset[:, :exp_dataset.observation_dim]
        target = exp_dataset.data[:,
                 exp_dataset.observation_dim + exp_dataset.action_dim:exp_dataset.observation_dim * 2 + exp_dataset.action_dim]
        Y_dataset = target - state
        
        self.Xm = np.atleast_1d(X_dataset.mean(0))
        # Xc = np.atleast_2d((np.cov(X_dataset - self.Xm, rowvar=False, ddof=1)))
        self.Xstd = np.atleast_1d(X_dataset.std(0))
        # self.iXs = np.linalg.cholesky(np.linalg.inv(Xc))
        
        self.Ym = np.atleast_1d(Y_dataset.mean(0))
        self.Ystd = np.atleast_1d(Y_dataset.std(0))
        # Yc = np.atleast_2d(np.cov(Y_dataset - self.Ym, rowvar=False, ddof=1))

        # self.Ys = np.linalg.cholesky(Yc).T
        
        # from numpy to torch
        self.Xm = Variable(torch.from_numpy(self.Xm).float().cuda())
        # self.iXs = Variable(torch.from_numpy(self.iXs).float().cuda())
        self.Ym = Variable(torch.from_numpy(self.Ym).float().cuda())
        # self.Ys = Variable(torch.from_numpy(self.Ys).float().cuda())
        self.Xstd = Variable(torch.from_numpy(self.Xstd).float().cuda())
        self.Ystd = Variable(torch.from_numpy(self.Ystd).float().cuda())

    def cost_fun(self, state, action, next_state):
        if self.env.spec.id == 'HalfCheetah2-v2' or self.env.spec.id == 'HalfCheetah-v2':
            cost = - (next_state[:, 17] - state[:, 17]) / 0.01 + 0.1 * torch.sum(torch.pow(action, 2), 1)
        elif self.env.spec.id == 'ArmReacherEnv-v0':
            cost = 0
            vec = state[:, 14:17] - state[:, 17:20]
            cost -= -torch.norm(vec) - 0.1 * torch.sum(torch.pow(action, 2), 1)
        elif self.env.spec.id == 'Ant2-v2' or self.env.spec.id == 'Ant-v2':
            cost = 0
            cost -= (next_state[:, 111] - state[:, 111]) / 0.01 - 0.5 * torch.sum(torch.pow(action, 2), 1) + 1
        elif self.env.spec.id == 'Swimmer2-v2' or self.env.spec.id == 'Swimmer-v2':
            cost = - (next_state[:, 8] - state[:, 8]) / 0.01 + 0.1 * torch.sum(torch.pow(action, 2), 1)
        elif self.env.spec.id == 'Hopper2-v2' or self.env.spec.id == 'Hopper-v2':
            cost = - (next_state[:, 11] - state[:, 11]) / 0.01 + 0.1 * torch.sum(torch.pow(action, 2), 1)
        else:
            assert print('Env Cost is not defined! ')
    
        return cost
   