import time

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from  kinematics.cheetah_kine_torch import fkine_pos

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


def cost_halfcheetah_com(states,   action=None, state_delat = False):
    """Pendulum-v0: Same as OpenAI-Gym"""
    #states = Variable(torch.zeros(21)).cuda()
    
   
    if not state_delat:
        com_p = states[0:3]
        com, foot1, foot2 = fkine_pos(com_p, states[3:9])
        
        com_d = Variable(torch.FloatTensor([0,0,0.6])).cuda()
        com_d[0] = states[-3]
        
        foot1_d = Variable(torch.FloatTensor([0,0, 0.02240554])).cuda()
        foot2_d = Variable(torch.FloatTensor([0, 0,0.05222651])).cuda()
        
        foot1_d[0] =states[-2]
        foot2_d[0] = states[-1]
        
        
        motor_cost = 0.01 * torch.norm(action)
        cost = torch.norm(com - com_d)+torch.norm(foot1 - foot1_d)+torch.norm(foot2 - foot2_d) #+ motor_cost
        
    else:
        foot2_delta = states[-3:]
        foot1_delta = states[-6:-3]
        com_delta =  states[-9:-6]
        
        
        motor_cost = 0.01 * torch.norm(action)
        cost = torch.norm(com_delta) + torch.norm(foot1_delta ) + torch.norm(foot2_delta ) + motor_cost
    return cost

 
 