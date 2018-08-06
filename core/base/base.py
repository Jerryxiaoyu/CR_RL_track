import time

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import torch.optim as optim
from core.shell.cost import cost, cost_halfcheetah, cost_halfcheetah_com
from core.utils import log
import matplotlib.pyplot as plt
#from MB.eval_model import plot_train, plot_train_std,load_data


PLOT_EVERY_N_EPOCH =10
LOG_EVERY_N_EPOCH =100

def rollout(env, policy,  max_steps=100 , init_particle=None, render=False ):
    """Generate one trajectory , return transitions
    
    data : D+E+1
    {state * D, action * E, cost *1 }
    
    """

    # Intial state
    if init_particle is not None:
        s = init_particle
        # Set Gym environment consistently
        env.reset()
        env.unwrapped.state = init_particle
    else:
        s = env.reset()

    policy.eval()
    data = []

    for _ in range(max_steps):
        if render:
            env.render()
        # Convert to FloatTensor, Variable and send to GPU
        s = Variable(torch.Tensor(s).unsqueeze(0)).cuda()
        # TODO Sometimes states need to be preprocessed!

        # Select an action by policy
        a = policy(s)
        # excuate action
        s_next, reward, done, _ = env.step(a.data.cpu().numpy())
        cost = -reward
        # Record data
 
        data.append(np.concatenate([s.data.cpu().numpy()[0], a.data.cpu().numpy().squeeze(), s_next, np.array([cost])]))
        # break if done
        if done:
            break
        # Update s as s_next for recording next transition
        s = s_next

    return np.array(data)

def gaussian_log_likelihood(targets, pred_mean, pred_std=None):
    ''' Computes the log likelihood for gaussian distributed predictions.
        This assumes diagonal covariances
    '''
    delta = pred_mean - targets
    # note that if we have nois be a 1xD vector, broadcasting
    # rules apply
    if pred_std:
        # sum over output dimensions
        lml = - torch.pow((delta/pred_std),2).sum(1)*0.5 - torch.log(pred_std).sum(1)
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
    eps =  np.finfo(np.__dict__['float64']).eps
    reg =[]
    for name, param in dynamics.named_parameters():
        reg_weight = 0.5
        if name.find('weight') != -1:
            reg_weight *= hidden_lengthscale ** 2
            print('reg_weight', reg_weight)
        
            p = 1 # 待修改
            
            W = param.data
            W_reg = reg_weight * torch.sum(p * W * W)
            p_reg = -torch.sum(p * torch.log(torch.Tensor([p + eps])))
            print(W_reg)
            print(p_reg)
            reg.append(W_reg + p_reg)
        if name.find('bias') != -1:
            b = param.data
            reg.append(reg_weight * torch.sum(b ** 2))

    return sum(reg)

def train_dynamics_model(dynamics, dynamics_optimizer, trainset, epochs=1, batch_size=1,**kwargs):
    # Create dynamics and its optimizer
    dynamics.set_sampling(sampling=False)

    log.infov('Dynamics training...')
    # Loss
    criterion = nn.MSELoss()  # MSE/SmoothL1

    dynamics.update_dataset_statistics(trainset)

    batch_size = trainset.data.shape[0] if trainset.data.shape[0]< batch_size else batch_size
    # Create Dataloader
    trainloader = Data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)

    (_, _), (x_test, y_test) = load_data()

    log.infov('Num of rollout: {} Data set size: {}'.format(len(trainset.buffer),trainset.data.shape[0]))
    dynamics.train()
    list_train_loss = []
    for epoch in range(epochs):  # Loop over dataset multiple times
        running_train_losses = []
        
        start_time = time.time()
        for i, data in enumerate(trainloader):  # Loop over batches of data
            # Get input batch
            X, Y = data

            
            
            # Wrap data tensors as Variable and send to GPU
            X = Variable(X).cuda()
            Y = Variable(Y).cuda()
            
            # Zero out the parameter gradients
            dynamics_optimizer.zero_grad()
            
            # Forward pass
            outputs = dynamics.predict_Y(X, delta_target=True, pre_prcess= kwargs['pre_process'])  # delta_target, return state difference for training

            
            
            # Loss
            loss = criterion(outputs, Y)
            M = Y.shape[0]
            N = X.shape[0]

            #lml = gaussian_log_likelihood(Y,outputs)

            reg = 0#  dropout_gp_kl(dynamics , input_lengthscale=1.0, hidden_lengthscale=1.0)
            
            #loss = -lml/M + reg/N
            # Backward pass
            loss.backward()

            # Update params
            dynamics_optimizer.step()
            
            # Accumulate running losses
            running_train_losses.append(loss.data[0])  # Take out value from 1D Tensor
        
        # Record the mean of training and validation losses in the batch
        batch_train_loss = np.mean(running_train_losses)
        list_train_loss.append(batch_train_loss)
        
        time_duration = time.time() - start_time
        # Logging: Only first, middle and last
        if epoch == 0 or epoch == epochs // 20 or epoch == epochs - 1:
            log.info('[Epoch # {:3d} ({:.1f} s)] Train loss: {:.8f}'.format(epoch + 1, time_duration, batch_train_loss))

        if epoch % LOG_EVERY_N_EPOCH == 0:
            eval_mse = plot_train(x_test, y_test, dyn_model=dynamics, pre_process=kwargs['pre_process'], plot=False)
            log.info('[Epoch # {:3d} ({:.1f} s)] Train loss: {:.8f} Eval loss: {:.8f}'.format(epoch + 1, time_duration,
                                                                                              batch_train_loss,
                                                                                              eval_mse))
        if epoch % PLOT_EVERY_N_EPOCH == 0:
            if kwargs['plot_train'] is not None:
                if callable(kwargs['plot_train']):
                    if epoch == 0:
                        
                        plt.ion()
                    kwargs['plot_train'](dynamics )
                
    log.info('Finished training dynamics model. ')
    return np.array(list_train_loss)


# (_, _), (x_test, y_test) = load_data()
# (_, _), (x_test2, y_test2) = load_data(
#     dir_name='/home/drl/PycharmProjects/DeployedProjects/deepPILCO/MB/data/log-test1.csv', data_num=1000)


def train_dynamics_model_pilco(dynamics, dynamics_optimizer, trainset, epochs=1, batch_size=1, eval_fn=None,logger=None ,**kwargs):
   
   # Create dynamics and its optimizer
    dynamics.set_sampling(sampling=False)
    
    log.infov('Dynamics training...')
    # Loss
    criterion = nn.MSELoss()  # MSE/SmoothL1
    
    dynamics.update_dataset_statistics(trainset)
    
    batch_size = trainset.data.shape[0] if trainset.data.shape[0] < batch_size else batch_size
    # Create Dataloader
    trainloader = Data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)


    
    log.infov('Num of rollout: {} Data set size: {}'.format(len(trainset.buffer), trainset.data.shape[0]))
    dynamics.train()
    list_train_loss = []
    for epoch in range(epochs):  # Loop over dataset multiple times
        running_train_losses = []
        
        start_time = time.time()
        for i, data in enumerate(trainloader):  # Loop over batches of data
            # Get input batch
            X, Y = data
            
            # Wrap data tensors as Variable and send to GPU
            X = Variable(X).cuda()
            Y = Variable(Y).cuda()
            
            # Zero out the parameter gradients
            dynamics_optimizer.zero_grad()
            
            # Forward pass
            outputs = dynamics.predict_Y(X, delta_target=True, pre_prcess=kwargs[
                'pre_process'])  # delta_target, return state difference for training
            
            # Loss
            loss = criterion(outputs, Y)
            M = Y.shape[0]
            N = X.shape[0]

           # loss = gaussian_log_likelihood(Y,outputs)
            
            reg = 0  # dropout_gp_kl(dynamics , input_lengthscale=1.0, hidden_lengthscale=1.0)

           # loss = -loss/M + reg/N
            # Backward pass
            loss.backward()
            
            # Update params
            dynamics_optimizer.step()
            
            # Accumulate running losses
            running_train_losses.append(loss.data[0])  # Take out value from 1D Tensor
        
        # Record the mean of training and validation losses in the batch
        batch_train_loss = np.mean(running_train_losses)
        list_train_loss.append(batch_train_loss)
        
        time_duration = time.time() - start_time
        # Logging: Only first, middle and last
    
        if epoch % LOG_EVERY_N_EPOCH == 0:
            
            if logger is not None:
                logger.log({'epoch': epoch,
                            'time_duration':time_duration,
                            'Train loss': batch_train_loss,
 
                            })
                logger.write(display=False)
            
        if epoch == 0 or epoch == epochs // 2 or epoch == epochs - 1:
            log.info(
                '[Epoch # {:3d} ({:.1f} s)] Train loss: {:.8f}  '.format(epoch + 1,
                                                                                                             time_duration,
                                                                                                             batch_train_loss))

        

    if logger is not None:
        logger.close()
    log.info('Finished training dynamics model. ')
    return np.array(list_train_loss)


def train_dynamics_model_pilco2(dynamics, dynamics_optimizer, trainset, epochs=1, batch_size=1, eval_fn=None,
                               logger=None, **kwargs):
    # Create dynamics and its optimizer
    dynamics.set_sampling(sampling=False)
    
    log.infov('Dynamics training...')
    # Loss
    #criterion = nn.MSELoss()  # MSE/SmoothL1
    
    dynamics.update_dataset_statistics(trainset)
    
    batch_size = trainset.data.shape[0] if trainset.data.shape[0] < batch_size else batch_size
    # Create Dataloader
    trainloader = Data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    (_, _), (x_test, y_test) = load_data()
    (_, _), (x_test2, y_test2) = load_data(
        dir_name='/home/drl/PycharmProjects/DeployedProjects/deepPILCO/MB/data/log-test1.csv', data_num=1000)
    
    log.infov('Num of rollout: {} Data set size: {}'.format(len(trainset.buffer), trainset.data.shape[0]))
    #dynamics.set_sampling(sampling= False)
    dynamics.train()
    list_train_loss = []
    for epoch in range(epochs):  # Loop over dataset multiple times
        running_train_losses = []
        
        start_time = time.time()
        for i, data in enumerate(trainloader):  # Loop over batches of data
            # Get input batch
            X, Y = data
            
            
            # Loss
            loss  = dynamics.get_loss( X, Y, pre_prcess = kwargs['pre_process'])
            
            
            # Backward pass
            loss.backward()
            
            # Update params
            dynamics_optimizer.step()
            
            # Accumulate running losses
            running_train_losses.append(loss.data[0])  # Take out value from 1D Tensor
        
        # Record the mean of training and validation losses in the batch
        batch_train_loss = np.mean(running_train_losses)
        list_train_loss.append(batch_train_loss)
        
        time_duration = time.time() - start_time
        # Logging: Only first, middle and last
        
        if epoch % LOG_EVERY_N_EPOCH == 0:
            if dynamics.env.spec.id == 'HalfCheetah-v2':
                eval_mse = plot_train(x_test, y_test, dyn_model=dynamics, pre_process=kwargs['pre_process'], plot=False)
                eval_mse2 = plot_train(x_test2, y_test2, dyn_model=dynamics, pre_process=kwargs['pre_process'],
                                       plot=False)
            # log.info('[Epoch # {:3d} ({:.1f} s)] Train loss: {:.8f} Eval loss1: {:.8f} Eval loss2: {:.8f}'.format(epoch + 1, time_duration, batch_train_loss, eval_mse, eval_mse2))
            else:
                eval_mse = 0
                eval_mse2 = 0
            if logger is not None:
                logger.log({'epoch': epoch,
                            'time_duration': time_duration,
                            'Train loss': batch_train_loss,
                            'Eval loss': eval_mse,
                            'Eval loss_export': eval_mse2,
                            })
                logger.write(display=False)
        
        if epoch == 0 or epoch == epochs // 2 or epoch == epochs - 1 or epoch % 5 == 0:
            log.info(
                '[Epoch # {:3d} ({:.1f} s)] Train loss: {:.8f} Eval loss1: {:.8f} Eval loss2: {:.8f}'.format(epoch + 1,
                                                                                                             time_duration,
                                                                                                             batch_train_loss,
                                                                                                             eval_mse,
                                                                                                             eval_mse2))
        
        if epoch % PLOT_EVERY_N_EPOCH == 0:
            if kwargs['plot_train'] is not None:
                if callable(kwargs['plot_train']):
                    if epoch == 0:
                        plt.ion()
                    kwargs['plot_train'](dynamics)
    
    if logger is not None:
        logger.close()
    log.info('Finished training dynamics model. ')
    return np.array(list_train_loss)
def learn_policy(env, dynamics, policy, policy_optimizer, K=1, T=1, gamma=0.99, init_particles=None,
                 moment_matching=True, c_sigma=0.25, grad_norm = None, pre_prcess=True):
    
    # Particles for initial state
    if init_particles is not None:
        particles = Variable(torch.Tensor(init_particles)).cuda()
    else:
        particles = Variable(torch.Tensor([env.reset() for _ in range(K)])).cuda()
    
    # Sample BNN dynamics: fixed dropout masks
    #dynamics.set_sampling(sampling=True, batch_size=K)

    dynamics.set_sampling(sampling=False )
    dynamics.train()

    policy.set_sampling(sampling=False)
    policy.train(mode=True)
    # List of costs
    list_costs = []
    # list of mu and sigma
    list_moments = []
    for t in range(T):  # time steps
      
      
        # K actions from policy given K particles
        actions = policy (particles )
        # Concatenate particles and actions as inputs to Dynamics model
        state_actions = torch.cat((particles, actions.view(-1,1)), 1)
        # Get next states from Bayesian Dynamics Model
        #next_states = dynamics(state_actions)
        next_states = dynamics.predict_Y(state_actions, delta_target=False, pre_prcess= pre_prcess)
        # Moment matching
        # Compute mean and standard deviation
        if moment_matching:
            mu = torch.mean(next_states, 0)
            sigma = torch.std(next_states, 0)
            # Standard normal noise for K particles
            z = Variable(torch.randn(K, sigma.size(0))).cuda()
            # Sample K new particles from a Gaussian by location-scale transformation/reparameterization
            particles = mu + sigma * z
            
            # Record mu and sigma
            list_moments.append([mu, sigma])
        else:
            particles = next_states
        
        # Compute the mean cost for the particles in the current time step
        # costs = torch.mean(cost(particles, sigma=c_sigma))
        costs = cost(torch.mean(particles, 0).unsqueeze(0), sigma=c_sigma)
        
        
        
        # Append the list of discounted costs
        list_costs.append((gamma ** (t + 1)) * costs)
    
    # Optimize policy
    policy_optimizer.zero_grad()
    # [optimizer.zero_grad() for optimizer in optimizers]
    J = torch.sum(torch.cat(list_costs))
    J.backward(retain_graph=True)

    # for policy in policies[1:]:
    #    for policy_param, cloned_param in zip(policies[0].parameters(), policy.parameters()):
    #        policy_param.grad.data += cloned_param.grad.data.clone()
    if grad_norm is not None:
        nn.utils.clip_grad_norm(policy.parameters(), grad_norm)
    # Original policy
    policy_optimizer.step()
    
    return policy, list_costs, list_moments


def learn_policy_pilco(env, dynamics, policy, policy_optimizer, K=1, T=1, gamma=0.99, init_particles=None,
                 moment_matching=True, c_sigma=0.25, grad_norm=None, pre_prcess=True, shaping_state_delta= False):
    # Particles for initial state
    if init_particles is not None:
        particles = Variable(torch.Tensor(init_particles)).cuda()
    else:
        particles = Variable(torch.Tensor([env.reset() for _ in range(K)])).cuda()

    shaping_state_constant = 9 if shaping_state_delta else 3
    
    goal = particles[:,-shaping_state_constant:]
    # Sample BNN dynamics: fixed dropout masks
    # dynamics.set_sampling(sampling=True, batch_size=K)
    
    dynamics.set_sampling(sampling=True,batch_size=K)
    dynamics.train()
    
    policy.set_sampling(sampling=True,batch_size=K)
    policy.train(mode=True)
    # List of costs
    list_costs = []
    # list of mu and sigma
    list_moments = []
    for t in range(T):  # time steps
        
        # K actions from policy given K particles
        policy.set_sampling(sampling=True, batch_size=K)
        actions = policy(particles)
        # Concatenate particles and actions as inputs to Dynamics model
        state_actions = torch.cat((particles, actions ), 1)
        # Get next states from Bayesian Dynamics Model
        # next_states = dynamics(state_actions)
        dynamics.set_sampling(sampling=True, batch_size=K)
        next_states = dynamics.predict_Y(state_actions, delta_target=False, pre_prcess=pre_prcess)
        # Moment matching
        # Compute mean and standard deviation
        if moment_matching:
            mu = torch.mean(next_states, 0)
            sigma = torch.std(next_states, 0)
            # Standard normal noise for K particles
            z = Variable(torch.randn(K, sigma.size(0))).cuda()
            # Sample K new particles from a Gaussian by location-scale transformation/reparameterization
            # TODO rewrite sigma
            particles_next = mu + sigma * Variable(torch.ones((K, sigma.size(0)))).cuda()
            
            # Record mu and sigma
            list_moments.append([mu, sigma])
        else:
            particles_next = next_states
        
        # Compute the mean cost for the particles in the current time step
        # costs = torch.mean(cost(particles, sigma=c_sigma))
        # costs = cost(torch.mean(particles, 0).unsqueeze(0), sigma=c_sigma)
        
      #  costs = cost_halfcheetah(torch.mean(particles_next,0).unsqueeze(0), torch.mean(particles,0).unsqueeze(0))
        costs = cost_halfcheetah_com( torch.mean(particles, 0), torch.mean(actions,0).unsqueeze(0), state_delat = shaping_state_delta)

 
        particles = torch.cat((particles_next,goal),1)
        
        # Append the list of discounted costs
        list_costs.append((gamma ** (t + 1)) * costs)
    
    # Optimize policy
    policy_optimizer.zero_grad()
    # [optimizer.zero_grad() for optimizer in optimizers]
    m = torch.cat(list_costs)
    J = torch.sum(m)
    J.backward(retain_graph=True)
    
    # for policy in policies[1:]:
    #    for policy_param, cloned_param in zip(policies[0].parameters(), policy.parameters()):
    #        policy_param.grad.data += cloned_param.grad.data.clone()
    if grad_norm is not None:
        nn.utils.clip_grad_norm(policy.parameters(), grad_norm)
    # Original policy
    policy_optimizer.step()
    
    return policy, list_costs, list_moments

def test_episodic_cost(env, policy,dynamics, N=1, T=1, render=False):
    ep_reward = []
    policy.eval()
    for _ in range(N):  # N episodes
        # Initial state
        s = env.reset()
        s = Variable(torch.Tensor(s).unsqueeze(0)).cuda()

        # Accumulated reward for current episode
        reward = []

        for t in range(T):  # T time steps
            # Xm = dynamics.Xm[:env.observation_space.shape[0]]
            # Xs = dynamics.Xstd[:env.observation_space.shape[0]]
            # s = (s - Xm) / Xs
            
            # Select action via policy
            a = policy(s).data.cpu().numpy()
            # Take action in the environment
            s_next, r, done, info = env.step(a)

            # Record reward
            reward.append(r)

            # Update new state
            s = Variable(torch.Tensor(s_next).unsqueeze(0)).cuda()

            if render:
                env.render()

        ep_reward.append(np.sum(reward))

    return -np.mean(ep_reward)/T, np.std(ep_reward)/T


def test_episodic_cost2(env, policy, dynamics=None, N=1, T=1, render=False):
    ep_reward = []
    policy.set_sampling(sampling=False )
    #policy.train(mode=True)
    policy.eval()
    for _ in range(N):  # N episodes
        # Initial state
        s = env.reset()
        s = Variable(torch.Tensor(s).unsqueeze(0)).cuda()
        
        # Accumulated reward for current episode
        reward = []
        
        for t in range(T):  # T time steps
            # Xm = dynamics.Xm[:env.observation_space.shape[0]]
            # Xs = dynamics.Xstd[:env.observation_space.shape[0]]
            # s = (s - Xm) / Xs

            # Select action via policy
            a = policy(s).data.cpu().numpy()[0]
            # Take action in the environment
            s_next, r, done, info = env.step(a)
            # Record reward
            reward.append(r)
            
            # Update new state
            s = Variable(torch.Tensor(s_next ).unsqueeze(0)).cuda()
            
            
            if render:
                env.render()
        
        ep_reward.append(np.sum(reward))
    
    return -np.mean(ep_reward) / N, np.std(ep_reward) / N