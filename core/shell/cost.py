import time

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

def cost(states, sigma=0.25):
    """Pendulum-v0: Same as OpenAI-Gym"""
    l = 0.6
    
    goal = Variable(torch.Tensor([0.0, l])).cuda()
    
    # Cart position
    cart_x = states[:, 0]
    # Pole angle
    thetas = states[:, 2]
    # Pole position
    x = torch.sin(thetas) * l
    y = torch.cos(thetas) * l
    positions = torch.stack([cart_x + x, y], 1)
    
    squared_distance = torch.sum((goal - positions) ** 2, 1)
    squared_sigma = sigma ** 2
    cost = 1 - torch.exp(-0.5 * squared_distance / squared_sigma)
    
    return cost


def cost_halfcheetah(states, last_states, action=None):
    """Pendulum-v0: Same as OpenAI-Gym"""

    sigma = 10

    squared_sigma = sigma ** 2
   

    speed =   (states[ :, 17] - last_states[:, 17]) / 0.01
    #cost = torch.exp(- 0.5 * speed / squared_sigma)
    cost = 1 - 1 / (1 + torch.exp(-(speed-5) * 1))
    
    return cost


def cost_halfcheetah_com(states, last_states, action=None):
    """Pendulum-v0: Same as OpenAI-Gym"""
    
    sigma = 10
    
    squared_sigma = sigma ** 2
    
    speed = (states[:, 17] - last_states[:, 17])
    

    # cost = torch.exp(- 0.5 * speed / squared_sigma)
    cost = 1 - 1 / (1 + torch.exp(-(speed - 5) * 1))
    
    return cost