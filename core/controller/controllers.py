
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import torch.nn.functional as F


class RandomPolicy(nn.Module):
    """Linear policy for the controller"""
    
    def __init__(self, env):
        super().__init__()
        
        self.env = env
    
    def forward(self, x):
        return Variable(torch.Tensor(self.env.action_space.sample())).cuda()


