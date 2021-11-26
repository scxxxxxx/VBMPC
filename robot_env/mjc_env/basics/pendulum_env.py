import numpy as np
from gym import utils
from os import path
from gym.envs.mujoco import mujoco_env
from robot_env.mjc_env.common import MujocoPyRenderer, CustomMjViewer, CustomMjEnv
import math
from mujoco_robot.basic_models.gen_model_pendulums import make_pendulum
from robot_env.utils import get_basic_model_path
import os
import copy


class Pendulum(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, n_joints, frame_skip=5, generate_xml=False):
        utils.EzPickle.__init__(**locals())
        print(n_joints, frame_skip)
        if generate_xml:
            model_path = os.path.join(get_basic_model_path(), "generated_pendulum{}.xml".format(n_joints))
            make_pendulum(n_joints)
        else:
            if n_joints == 2:
                model_path = "double_pendulum.xml"
            elif n_joints == 3:
                model_path = "triple_pendulum.xml"
            elif n_joints == 4:
                model_path = "quadruple_pendulum.xml"
            else:
                raise NotImplementedError
            model_path = os.path.join(get_basic_model_path(), model_path)
        self.frame_skip = frame_skip
        mujoco_env.MujocoEnv.__init__(self, model_path, frame_skip=self.frame_skip)

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
        self.viewer.cam.distance = self.model.stat.extent * 2.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20
