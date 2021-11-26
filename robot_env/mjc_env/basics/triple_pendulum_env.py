import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import math
import os
import copy


class TriplePendulum(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, frame_skip=5):
        utils.EzPickle.__init__(**locals())
        module_path = os.path.dirname(__file__)
        model_path = "robots/basic_models/triple_pendulum.xml"
        full_path = os.path.abspath(os.path.join(module_path, '../..', model_path))
        self.frame_skip = frame_skip
        mujoco_env.MujocoEnv.__init__(self, full_path, frame_skip=self.frame_skip)

    def _get_obs(self):
        data = self.sim.data
        qpos = data.qpos.flat.copy()
        qvel = data.qvel.flat.copy()
        cinert = data.cinert.flat.copy()
        cvel = data.cvel.flat.copy()
        qfrc_actuator = data.qfrc_actuator.flat.copy()
        cfrc_ext = data.cfrc_ext.flat.copy()
        return np.concatenate([qpos, qvel, cinert, cvel, qfrc_actuator, cfrc_ext])

    def get_image(self, camera_name=None, shape=(200, 200), depth=False):
        im = copy.deepcopy(self.sim.render(width=shape[0], height=shape[1], camera_name=camera_name, depth=depth))
        # im[0]: rgb image, im[1]: depth image
        return im

    @property
    def done(self):
        # implement the custom done property
        done = False
        return done

    @property
    def reward(self):
        # implement the reward
        reward = 0
        return reward

    def step(self, a):
        # a = np.zeros_like(a)
        # print(a)
        self.do_simulation(a, self.frame_skip)
        return self._get_obs(), self.reward, self.done, None
        # None could be a dict(reward_linup=uph_cost, reward_quadctrl=-quad_ctrl_cost, reward_impact=-quad_impact_cost)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        # self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20
