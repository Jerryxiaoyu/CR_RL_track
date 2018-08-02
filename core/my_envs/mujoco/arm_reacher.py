import numpy as np
from gym import utils
from core.my_envs.mujoco import mujoco_env

import mujoco_py

class ArmReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'armreacher.xml', 5)

    def step(self, a):
        

        vec = self.get_body_com("tips_arm") - self.get_body_com("goal")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + 0.1 * reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        
        return ob, reward, done, dict(reward_dist=reward_dist,
                reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self,reset_args= None):
        qpos = self.init_qpos

        while True:
            self.goal_pos = np.concatenate([
                    self.np_random.uniform(low=-0.5, high=0.5, size=1),
                    self.np_random.uniform(low= -0.5, high=-0.2, size=1),
                    self.np_random.uniform(low= 0.2, high=0.5, size=1)])
            if np.linalg.norm(self.goal_pos) < 2:
                break
        if reset_args == 'fixed':
            qpos[-3:] = np.array([0.3, -0.3, 0.3])
        else:
            qpos[-3:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                high=0.005, size=self.model.nv)
        qvel[-3:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[:7],
            self.sim.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
 
            self.get_body_com("goal"),
        ])
