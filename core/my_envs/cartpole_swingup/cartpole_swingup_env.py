"""
Cart pole swing-up: Identical version to PILCO V0.9
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)



# class CartPoleSwingUpEnv(gym.Env):
#     metadata = {
#         'render.modes': ['human', 'rgb_array'],
#         'video.frames_per_second' : 50
#     }

#     def __init__(self):
#         self.g = 9.82  # gravity
#         self.m_c = 0.5  # cart mass
#         self.m_p = 0.5  # pendulum mass
#         self.total_m = (self.m_p + self.m_c)
#         self.l = 0.6 # pole's length
#         self.m_p_l = (self.m_p*self.l)
#         self.force_mag = 0.01
#         self.dt = 0.05  # seconds between state updates
#         self.b = 0.1  # friction coefficient

#         # Angle at which to fail the episode
#         self.theta_threshold_radians = 12 * 2 * math.pi / 360
#         self.x_threshold = 2.4

#         high = np.array([
#             np.finfo(np.float32).max,
#             np.finfo(np.float32).max,
#             np.finfo(np.float32).max,
#             np.finfo(np.float32).max])

#         self.action_space = spaces.Box(-self.force_mag, self.force_mag, shape=(1,))
#         self.observation_space = spaces.Box(-high, high)

#         self.seed()
#         self.viewer = None
#         self.state = None

#     def seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]

#     def step(self, action):
#         # Valid action
#         action = np.clip(action, -self.force_mag, self.force_mag)[0]

#         state = self.state
#         x, x_dot, theta, theta_dot = state
        
#         s = math.sin(theta)
#         c = math.cos(theta)
        
#         xdot_update = (-2*self.m_p_l*(theta_dot**2)*s + 3*self.m_p*self.g*s*c + 4*action - 4*self.b*x_dot)/(4*self.total_m - 3*self.m_p*c**2)
#         thetadot_update = (-3*self.m_p_l*(theta_dot**2)*s*c + 6*self.total_m*self.g*s + 6*(action - self.b*x_dot)*c)/(4*self.l*self.total_m - 3*self.m_p_l*c**2)
#         x = x + x_dot*self.dt
#         theta = theta + theta_dot*self.dt
#         x_dot = x_dot + xdot_update*self.dt
#         theta_dot = theta_dot + thetadot_update*self.dt
        
#         self.state = (x,x_dot,theta,theta_dot)
        
#         # compute costs - saturation cost
#         goal = np.array([0.0, self.l])
#         pole_x = self.l*np.sin(theta)
#         pole_y = self.l*np.cos(theta)
#         position = np.array([self.state[0] + pole_x, pole_y])
#         squared_distance = np.sum((position - goal)**2)
#         squared_sigma = 0.5**2
#         costs = 1 - np.exp(-0.5*squared_distance/squared_sigma)
        
#         return np.array(self.state), -costs, False, {}

#     def reset(self):
#         #self.state = self.np_random.normal(loc=np.array([0.0, 0.0, 30*(2*np.pi)/360, 0.0]), scale=np.array([0.0, 0.0, 0.0, 0.0]))
#         self.state = self.np_random.normal(loc=np.array([0.0, 0.0, np.pi, 0.0]), scale=np.array([0.02, 0.02, 0.02, 0.02]))
#         self.steps_beyond_done = None
#         return np.array(self.state)

#     def render(self, mode='human', close=False):
#         if close:
#             if self.viewer is not None:
#                 self.viewer.close()
#                 self.viewer = None
#             return

#         screen_width = 600
#         screen_height = 400

#         world_width = 5  # max visible position of cart
#         scale = screen_width/world_width
#         carty = 200 # TOP OF CART
#         polewidth = 6.0
#         polelen = scale*self.l  # 0.6 or self.l
#         cartwidth = 40.0
#         cartheight = 20.0

#         if self.viewer is None:
#             from gym.envs.classic_control import rendering
#             self.viewer = rendering.Viewer(screen_width, screen_height)
            
#             l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            
#             cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
#             self.carttrans = rendering.Transform()
#             cart.add_attr(self.carttrans)
#             cart.set_color(1, 0, 0)
#             self.viewer.add_geom(cart)
            
#             l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
#             pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
#             pole.set_color(0, 0, 1)
#             self.poletrans = rendering.Transform(translation=(0, 0))
#             pole.add_attr(self.poletrans)
#             pole.add_attr(self.carttrans)
#             self.viewer.add_geom(pole)
            
#             self.axle = rendering.make_circle(polewidth/2)
#             self.axle.add_attr(self.poletrans)
#             self.axle.add_attr(self.carttrans)
#             self.axle.set_color(0.1, 1, 1)
#             self.viewer.add_geom(self.axle)
            
#             # Make another circle on the top of the pole
#             self.pole_bob = rendering.make_circle(polewidth/2)
#             self.pole_bob_trans = rendering.Transform()
#             self.pole_bob.add_attr(self.pole_bob_trans)
#             self.pole_bob.add_attr(self.poletrans)
#             self.pole_bob.add_attr(self.carttrans)
#             self.pole_bob.set_color(0, 0, 0)
#             self.viewer.add_geom(self.pole_bob)
            
            
#             self.wheel_l = rendering.make_circle(cartheight/4)
#             self.wheel_r = rendering.make_circle(cartheight/4)
#             self.wheeltrans_l = rendering.Transform(translation=(-cartwidth/2, -cartheight/2))
#             self.wheeltrans_r = rendering.Transform(translation=(cartwidth/2, -cartheight/2))
#             self.wheel_l.add_attr(self.wheeltrans_l)
#             self.wheel_l.add_attr(self.carttrans)
#             self.wheel_r.add_attr(self.wheeltrans_r)
#             self.wheel_r.add_attr(self.carttrans)
#             self.wheel_l.set_color(0, 0, 0)  # Black, (B, G, R)
#             self.wheel_r.set_color(0, 0, 0)  # Black, (B, G, R)
#             self.viewer.add_geom(self.wheel_l)
#             self.viewer.add_geom(self.wheel_r)
            
#             self.track = rendering.Line((0,carty - cartheight/2 - cartheight/4), (screen_width,carty - cartheight/2 - cartheight/4))
#             self.track.set_color(0,0,0)
#             self.viewer.add_geom(self.track)

#         if self.state is None: return None

#         x = self.state
#         cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
#         self.carttrans.set_translation(cartx, carty)
#         self.poletrans.set_rotation(x[2])
#         self.pole_bob_trans.set_translation(-self.l*np.sin(x[2]), self.l*np.cos(x[2]))

#         return self.viewer.render(return_rgb_array = mode=='rgb_array')

"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

class CartPoleSwingUpEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        
        self.gravity = 9.82
        self.masscart = 0.5
        self.masspole = 0.5
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.b = 0.1  # friction coefficient

        self.tau = 0.1  # seconds between state updates

        self.sigma_c = 0.5

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.force_mag = 10.0
        self.action_space = spaces.Box(-np.array([self.force_mag]), np.array([self.force_mag]))
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        state = self.state
        x, x_dot, theta, theta_dot = state

        force = np.clip(action, -self.force_mag, self.force_mag)[0]

        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x,x_dot,theta,theta_dot)

        # compute costs - saturation cost
        goal = np.array([0.0, self.length])
        pole_x = self.length*np.sin(theta)
        pole_y = self.length*np.cos(theta)
        position = np.array([self.state[0] + pole_x, pole_y])
        squared_distance = np.sum((position - goal)**2)
        squared_sigma = self.sigma_c**2
        cost = 1 - np.exp(-0.5*squared_distance/squared_sigma)

        done = False

        return np.array(self.state), -cost, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(4,))
        self.state[2] = self.np_random.uniform(low=math.pi-0.2, high=math.pi+0.2, size=(1,))
        # print(self.state)
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self,mode='human', close=False):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()
