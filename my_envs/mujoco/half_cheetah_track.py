import numpy as np
from gym import utils
from my_envs.mujoco import mujoco_env
from kinematics.cheetah_kine import desired_Body, fkine_pos
import math


# goal_switch_time
switch_time = 10.0
class HalfCheetahTrackEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.delta = False
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 1)
        utils.EzPickle.__init__(self)
        
    def step(self, action):
        n_step = math.ceil(self.sim.data.time/self.dt)
        if n_step % int(switch_time/self.dt) == 0:
            self.com_p_d, self.foot1_d, self.foot2_d = desired_Body(self.sim.data.time, switch_time)
            # print('goal')
            # print(self.com_p_d, self.foot1_d,self.foot2_d)
            self.sim.model.site_pos[3] = np.array(self.com_p_d).reshape((1, -1))[0]
            self.sim.model.site_pos[4] = np.array(self.foot1_d).reshape((1, -1))[0]
            self.sim.model.site_pos[5] = np.array(self.foot2_d).reshape((1, -1))[0]
            
        self.do_simulation(action, self.frame_skip)
        
        ob = self._get_obs()
        com_p = ob[0:3]
        com, foot1, foot2 = fkine_pos(com_p, ob[3:9])

        self.sim.model.site_pos[0] = np.array(com).reshape((1,-1))[0]
        self.sim.model.site_pos[1] = np.array(foot1).reshape((1,-1))[0]
        self.sim.model.site_pos[2] = np.array(foot2).reshape((1,-1))[0]
        
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = -np.linalg.norm(com-self.com_p_d) - np.linalg.norm(foot1 - self.foot1_d) -np.linalg.norm(foot2-self.foot2_d)
        reward = reward_ctrl + reward_run
        
        #print(reward_ctrl, reward_run, reward)
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        if self.delta:
            com_cal, foot1_cal, foot2_cal = fkine_pos(self.sim.data.qpos[0:3],self.sim.data.qpos[3:9])
    
            com  = self.com_p_d  - com_cal
            foot1 = self.foot1_d -foot1_cal
            foot2 = self.foot2_d -foot2_cal
        else:
            com = self.com_p_d[0]
            foot1 = self.foot1_d[0]
            foot2 = self.foot2_d[0]
        
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            #self.get_body_com("torso").flat,
            com.flat,
            foot1.flat,
            foot2.flat
        ])

    def reset_model(self,reset_args=None):
        qpos = self.init_qpos  + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel  + self.np_random.randn(self.model.nv) * .1
        
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
