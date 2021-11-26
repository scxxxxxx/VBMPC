
from gym.envs.mujoco import mujoco_env
from gym import utils
import numpy as np
import os
import inspect


class CassieEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(**locals())
        module_path = os.path.dirname(__file__)
        model_path = "robots/cassie-mujoco/robot/cassie.xml"
        full_path = os.path.abspath(os.path.join(module_path, '../..', model_path))
        # print("load robot model from path: ", full_path)
        self.frame_skip = 5
        mujoco_env.MujocoEnv.__init__(self, full_path, frame_skip=self.frame_skip)
        # self.width = None
        # self.height = None
        # self.camera_id = None
        # self.camera_name = None

    def _get_obs(self):
        data = self.sim.data
        qpos = data.qpos.flat.copy()
        qvel = data.qvel.flat.copy()
        cinert = data.cinert.flat.copy()
        cvel = data.cvel.flat.copy()
        qfrc_actuator = data.qfrc_actuator.flat.copy()
        cfrc_ext = data.cfrc_ext.flat.copy()
        return np.concatenate([qpos, qvel, cinert, cvel, qfrc_actuator, cfrc_ext])

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
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 0.8925
        self.viewer.cam.elevation = -20
