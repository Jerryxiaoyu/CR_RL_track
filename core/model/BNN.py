import numpy as np

import torch

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import torch.nn.functional as F


class BNN(nn.Module):
    """Learning dynamics model via regression"""
    
    def __init__(self, env, hidden_size=200, drop_prob=0.0):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob
        
        # Flag for sampling parameters
        self.sampling = False
        # Fix the random mask for dropout, each batch contains K particles
        self.mask1 = None
        self.mask2 = None
        
        # Fully connected layer
        self.fc1 = nn.Linear(in_features=env.observation_space.shape[0] + env.action_space.shape[0],  # State + Action
                             out_features=self.hidden_size,
                             bias=True)
        self.fc2 = nn.Linear(in_features=self.hidden_size,
                             out_features=self.hidden_size,
                             bias=True)
        self.out = nn.Linear(in_features=self.hidden_size,
                             out_features=env.observation_space.shape[0],  # Next state
                             bias=True)
    
    def forward(self, x, delta_target=False, training=True):
        # Check if drop mask with correct dimension
        if self.sampling:
            if self.mask1.size()[0] != x.size()[0] or self.mask2.size()[0] != x.size()[0]:
                raise ValueError('Dimension of fixed masks must match the batch size.')
        state = x.clone()[:, :-1]  # CartPoleSwingUp, without action
        x = F.relu(self.fc1(x))  # try sigmoid as DeepPILCO paper
        if self.sampling:
            x = x * self.mask1
        else:
            x = F.dropout(x, p=self.drop_prob, training=training)
        x = F.relu(self.fc2(x))  # try sigmoid as DeepPILCO paper
        if self.sampling:
            x = x * self.mask2
        else:
            x = F.dropout(x, p=self.drop_prob, training=training)
        x = self.out(x)
        if delta_target:  # return difference in states, for training
            x = x
        else:  # return next states as s + delta_s
            x = state + x
        return x
    
    def set_sampling(self, sampling=None, batch_size=None):
        if sampling is None:
            raise ValueError('Sampling cannot be None.')
        
        self.sampling = sampling
        
        if self.sampling:
            # Sample dropout random masks
            self.mask1 = Variable(
                torch.bernoulli(torch.zeros(batch_size, self.hidden_size).fill_(1 - self.drop_prob))).cuda()
            self.mask2 = Variable(
                torch.bernoulli(torch.zeros(batch_size, self.hidden_size).fill_(1 - self.drop_prob))).cuda()
            # Rescale by 1/p to maintain output magnitude
            self.mask1 /= (1 - self.drop_prob)
            self.mask2 /= (1 - self.drop_prob)


def dropout(p=None, dim=None, method='standard'):
    if method == 'standard':
        return nn.Dropout(p)
    elif method == 'gaussian':
        return GaussianDropout(p / (1 - p))
    elif method == 'variational':
        return VariationalDropout(p / (1 - p), dim)


class BNN2(nn.Module):
    """Learning dynamics model via regression"""
    
    def __init__(self, env, hidden_size=200, drop_prob=0.0, dropout_method='standard'):
        super(BNN2, self).__init__()
        
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob
        
        # Flag for sampling parameters
        self.sampling = False
        # Fix the random mask for dropout, each batch contains K particles
        self.dropout_method = dropout_method
        
        self.input_dim = env.observation_space.shape[0] + env.action_space.shape[0]
        self.output_dim = env.observation_space.shape[0]
        # Fully connected layer
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            dropout(self.drop_prob, self.hidden_size, self.dropout_method),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            dropout(self.drop_prob, self.hidden_size, self.dropout_method),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_dim)
        )
    
    def kl(self):
        kl = 0
        for name, module in self.net.named_modules():
            if isinstance(module, VariationalDropout):
                kl += module.kl().sum()
        return kl
    
    def forward(self, x, delta_target=True):
        return self.net(x)


class VariationalDropout(nn.Module):
    def __init__(self, alpha=1.0, dim=None):
        super(VariationalDropout, self).__init__()
        
        self.dim = dim
        self.max_alpha = alpha
        # Initial alpha
        log_alpha = (torch.ones(dim) * alpha).log()
        self.log_alpha = nn.Parameter(log_alpha)
    
    def kl(self):
        c1 = 1.16145124
        c2 = -1.50204118
        c3 = 0.58629921
        
        alpha = self.log_alpha.exp()
        
        negative_kl = 0.5 * self.log_alpha + c1 * alpha + c2 * alpha ** 2 + c3 * alpha ** 3
        
        kl = -negative_kl
        
        return kl.mean()
    
    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(0,1)
            epsilon = Variable(torch.randn(x.size()))
            if x.is_cuda:
                epsilon = epsilon.cuda()
            
            # Clip alpha
            self.log_alpha.data = torch.clamp(self.log_alpha.data, max=self.max_alpha)
            alpha = self.log_alpha.exp()
            
            # N(1, alpha)
            epsilon = epsilon * alpha
            
            return x * epsilon
        else:
            return x


class GaussianDropout(nn.Module):
    def __init__(self, alpha=1.0):
        super(GaussianDropout, self).__init__()
        self.alpha = torch.Tensor([alpha])
    
    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(1, alpha)
            epsilon = torch.randn(x.size()) * self.alpha + 1
            
            epsilon = Variable(epsilon)
            if x.is_cuda:
                epsilon = epsilon.cuda()
            
            return x * epsilon
        else:
            return x


class BNN3(nn.Module):
    """
    Learning dynamics model via regression
    
    fro complex module
    
    """
    
    def __init__(self, env, hidden_size=[200]*3, drop_prob=0.0, activation='relu', shaping_state_delta = False):
        super().__init__()
        
        
        self.env = env
        # Flag for sampling parameters
        self.sampling = False
        # Fix the random mask for dropout, each batch contains K particles
        self.mask = []
        self.hidden_size = hidden_size

        self.shaping_state_delta = shaping_state_delta
        self. shaping_state_constant = 9 if self.shaping_state_delta else 3
        
        for i in range(len(hidden_size)):
            self.mask.append(None)
        self.obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.input_dim = self.obs_dim + act_dim
        self.ouput_dim = self.obs_dim - self.shaping_state_constant
        
        
        
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
    
    def forward(self, x, training=True,delta_target=True):
        
        
        state = x.clone()[:, :self.obs_dim]
        if self.sampling:
         
            for i,affine in enumerate(self.fc_layers):
                # Check if drop mask with correct dimension
                if self.mask[i].size()[0] != x.size()[0]  :
                    raise ValueError('Dimension of fixed masks must match the batch size.')
                x = self.activation(affine(x))
                x = x * self.mask[i]
        else:
            for affine in self.fc_layers:
                x = self.activation(affine(x))
                x = self.dropout(x )
        
        x = self.out_layer(x)
        
        if delta_target:
            x = x
        else:
            x = state + x
        
        return x
    
    def set_sampling(self, sampling=None, batch_size=None):
        if sampling is None:
            raise ValueError('Sampling cannot be None.')
        
        self.sampling = sampling
        
        if self.sampling:
            # Sample dropout random masks
            
            for i,affine in enumerate(self.fc_layers):
                self.mask[i] = Variable(
                torch.bernoulli(torch.zeros(batch_size, self.hidden_size[i]).fill_(1 - self.drop_prob))).cuda()
                # Rescale by 1/p to maintain output magnitude
                self.mask[i]/= (1 - self.drop_prob)
          
    
    def _set_init_param(self):
        for param in self.parameters():
            nn.init.normal(param, mean=0, std=1e-2)
    def predict_Y(self, x , delta_target = True, pre_prcess = True):
        state = x.clone()[:, :self.obs_dim- self.shaping_state_constant]
        if pre_prcess:
            # standardize inputs
            #x = torch.matmul(x - self.Xm, self.iXs)
            x = (x - self.Xm)/self.Xstd

        # Forward pass
        y = self.forward(x, delta_target=True)

        if pre_prcess:
            # scale and center outputs
            #y= torch.matmul(y, self.Ys) +self.Ym
            y = y * self.Ystd[:self.obs_dim- self.shaping_state_constant] + self.Ym[:self.obs_dim- self.shaping_state_constant]

        if delta_target:
            y = y
        else:
            y = y + state
        
        return y
        
        
    def predict(self,state, action):
    
        x_data = torch.cat((state, action), 1)
       # x_data = (x_data - self.scaler_cuda_x_mean) / self.scaler_cuda_x_scaler
    
        x_data = Variable(x_data)
    
        f = self.forward(x_data)
       # f = f.data * self.scaler_cuda_y_scaler + self.scaler_cuda_y_mean
        next_states = f.data + state
    
        return next_states
    def predict_samples(self, x , delta_target = True, pre_prcess = True ):
    
        state = x.clone()[:, :self.obs_dim- self.shaping_state_constant]
        action = x.clone()[:,self.obs_dim :]
        if pre_prcess:
            # standardize inputs
            x = (x - self.Xm) / self.Xstd
    
        # Forward pass
        y = self.forward(x, delta_target=True)
    
        if pre_prcess:
            # scale and center outputs
            y = y * self.Ystd + self.Ym
    
        if delta_target:
            y = y
        else:
            y = y + state
         
        
        #cost = - (y[:,17] - state[:,17]) / 0.01 + 0.1 * torch.sum(torch.pow(action,2), 1)
        cost = self.cost_fun(state, action, y)
        return y, cost

    def update_dataset_statistics(self, exp_dataset):
    
        X_dataset = exp_dataset.data[:, :exp_dataset.observation_dim + exp_dataset.action_dim]
        X_dataset += 1e-6 * np.random.randn(*X_dataset.shape)
        state = X_dataset[:, :exp_dataset.observation_dim]
        target = exp_dataset.data[:, exp_dataset.observation_dim + exp_dataset.action_dim:exp_dataset.observation_dim * 2 + exp_dataset.action_dim]
        Y_dataset = target - state
    
        self.Xm = np.atleast_1d(X_dataset.mean(0))
        #Xc = np.atleast_2d((np.cov(X_dataset - self.Xm, rowvar=False, ddof=1)))
        self.Xstd = np.atleast_1d(X_dataset.std(0))
        #self.iXs = np.linalg.cholesky(np.linalg.inv(Xc))
    
        self.Ym = np.atleast_1d(Y_dataset.mean(0))
        self.Ystd = np.atleast_1d(Y_dataset.std(0))
        #Yc = np.atleast_2d(np.cov(Y_dataset - self.Ym, rowvar=False, ddof=1))
    
        #self.Ys = np.linalg.cholesky(Yc).T

        # from numpy to torch
        self.Xm = Variable(torch.from_numpy(self.Xm).float().cuda())
        #self.iXs = Variable(torch.from_numpy(self.iXs).float().cuda())
        self.Ym = Variable(torch.from_numpy(self.Ym).float().cuda())
        #self.Ys = Variable(torch.from_numpy(self.Ys).float().cuda())
        self.Xstd = Variable(torch.from_numpy(self.Xstd).float().cuda())
        self.Ystd = Variable(torch.from_numpy(self.Ystd).float().cuda())
    def cost_fun(self, state, action, next_state):
        if self.env.spec.id == 'HalfCheetah2-v2' or self.env.spec.id == 'HalfCheetah-v2':
            cost = - (next_state[:, 17] - state[:, 17]) / 0.01 + 0.1 * torch.sum(torch.pow(action, 2), 1)
        elif self.env.spec.id == 'ArmReacherEnv-v0':
            cost =0
            vec = state[:, 14:17] - state[:, 17:20]
            cost -= -torch.norm(vec) - 0.1 * torch.sum(torch.pow(action, 2), 1)
        elif self.env.spec.id == 'Ant2-v2'or self.env.spec.id == 'Ant-v2':
            cost = 0
            cost -= (next_state[:, 111] - state[:, 111]) / 0.01 - 0.5 * torch.sum(torch.pow(action, 2), 1) + 1
        elif self.env.spec.id == 'Swimmer2-v2' or self.env.spec.id == 'Swimmer-v2' :
            cost = - (next_state[:, 8] - state[:, 8]) / 0.01 + 0.1 * torch.sum(torch.pow(action, 2), 1)
        elif self.env.spec.id == 'Hopper2-v2' or self.env.spec.id == 'Hopper-v2' :
            cost = - (next_state[:, 11] - state[:, 11]) / 0.01 + 0.1 * torch.sum(torch.pow(action, 2), 1)
        else:
            assert print('Env Cost is not defined! ')
            
        return cost