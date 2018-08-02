import numpy as np

import torch

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import torch.nn.functional as F

def gaussian_log_likelihood(targets, pred_mean, pred_std=None):
    ''' Computes the log likelihood for gaussian distributed predictions.
        This assumes diagonal covariances
    '''
    delta = pred_mean - targets
    # note that if we have nois be a 1xD vector, broadcasting
    # rules apply
    if pred_std is not None:
        # sum over output dimensions
        lml = - torch.div(torch.pow((delta),2), pred_std).sum(1)*0.5 - 0.5*torch.log(pred_std).sum(1)
    else:
        # sum ove output dimensions
        lml = - torch.pow((delta ),2).sum(1)*0.5

    # sum over all examples
    return lml.sum()


def dropout_gp_kl(dynamics, input_lengthscale=1.0, hidden_lengthscale=1.0):
    '''
        KL divergence approximation for the dropout uncertainty model from
        Gal and Ghahrammani, 2015
    '''
    eps = np.finfo(np.__dict__['float64']).eps
    reg = []
    for name, param in dynamics.named_parameters():
       # print('name :', name)
        reg_weight = 0.5
        if name.find('weight') != -1:
            reg_weight *= hidden_lengthscale ** 2
           # print('reg_weight: ', reg_weight)
            if name.find('out') != -1:
                p=1
            else:
                p = 1-dynamics.drop_prob  # 待修改
           # print('p:',p)
            W = param
            W_reg = reg_weight * torch.sum(p * W * W)
            p_reg = -torch.sum(p * torch.log(Variable(torch.Tensor([p + eps])))).cuda()
            #print('W_reg:', W_reg)
           # print('p_reg: ',p_reg)
            reg.append(W_reg + p_reg)
        if name.find('bias') != -1:
            b = param.data
            reg.append(reg_weight * torch.sum(b ** 2))
    
    return sum(reg)

class BNN3_h(nn.Module):
    """
    Learning dynamics model via regression
    
    fro complex module
    
    """
    
    def __init__(self, env, hidden_size=[200]*3, drop_prob=0.0, activation='relu'):
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
        self.input_dim = self.obs_dim + self.act_dim
        self.ouput_dim = self.obs_dim *2
        
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

        self.likelihood = gaussian_log_likelihood

        
    
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
        state = x.clone()[:, :self.obs_dim]
        if pre_prcess:
            # standardize inputs
            x = torch.matmul(x - self.Xm, self.iXs)

        # Forward pass
        res = self.forward(x, delta_target=True)

        y = res[:, :self.obs_dim]
        sn = 0.1 * torch.sigmoid(res[:, self.obs_dim:])
        # fudge factor
        sn += 1e-6

        if pre_prcess:
            # scale and center outputs
            y = torch.matmul(y, self.Ys) + self.Ym
            #            y = y.dot(self.Ys) + self.Ym
            # rescale variances
            sn = sn * torch.diag(self.Ys)

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
    
        state = x.clone()[:, :self.obs_dim]
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
        Xc = np.atleast_2d((np.cov(X_dataset - self.Xm, rowvar=False, ddof=1)))
        self.Xstd = np.atleast_1d(X_dataset.std(0))
        self.iXs = np.linalg.cholesky(np.linalg.inv(Xc))
    
        self.Ym = np.atleast_1d(Y_dataset.mean(0))
        self.Ystd = np.atleast_1d(Y_dataset.std(0))
        Yc = np.atleast_2d(np.cov(Y_dataset - self.Ym, rowvar=False, ddof=1))
    
        self.Ys = np.linalg.cholesky(Yc).T

        # from numpy to torch
        self.Xm = Variable(torch.from_numpy(self.Xm).float().cuda())
        self.iXs = Variable(torch.from_numpy(self.iXs).float().cuda())
        self.Ym = Variable(torch.from_numpy(self.Ym).float().cuda())
        self.Ys = Variable(torch.from_numpy(self.Ys).float().cuda())
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
    
    def get_loss(self, X,Y, pre_prcess):
        x = Variable(X).cuda()
        y = Variable(Y).cuda()

        if pre_prcess:
            # standardize inputs
            x = torch.matmul(x - self.Xm, self.iXs)
            
        # Forward pass
        res = self.forward(x, delta_target=True)
        
        
        y_predict =  res[:, :self.obs_dim]
        sn = 0.1 * torch.sigmoid(res[:, self.obs_dim:])
        # fudge factor
        sn += 1e-6
        
        if pre_prcess:
            # scale and center outputs
            y= torch.matmul(y_predict, self.Ys) +self.Ym
#            y = y.dot(self.Ys) + self.Ym
            # rescale variances
            sn = sn * torch.diag(self.Ys)

        M = y.shape[0]
        N = x.shape[0]   # TODO 貌似是总数据集的尺寸
        
        lml = self.likelihood(y, y_predict, sn)

        reg =   dropout_gp_kl(self , input_lengthscale=1.0, hidden_lengthscale=1.0)
        
        
        loss =  -lml / M + reg / N
        
        return loss

         