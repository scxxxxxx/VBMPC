from gym.envs.mujoco import mujoco_env
from gym import utils
import numpy as np

from robot_env.utils import get_robot_path


class Armar4StandingEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(**locals())
        self.frame_skip = 5
        self.root = "root"
        self.lhtcp = "HandL_TCP"
        self.rhtcp = "HandR_TCP"
        self.lltcp = "LegL_TCP"
        self.rltcp = "LegR_TCP"

        model_path = get_robot_path("armar4-mujoco/environment/Armar4_leg_only.xml", "mujoco")
        mujoco_env.MujocoEnv.__init__(self, model_path, frame_skip=self.frame_skip)
        # self.width = None
        # self.height = None
        # self.camera_id = None
        # self.camera_name = None

    def _get_obs(self):
        data = self.sim.data
        qpos = data.qpos.flat.copy()
        qvel = data.qvel.flat.copy()
        # cinert = data.cinert.flat.copy()
        # cvel = data.cvel.flat.copy()
        # qfrc_actuator = data.qfrc_actuator.flat.copy()
        # cfrc_ext = data.cfrc_ext.flat.copy()
        return np.concatenate([qpos, qvel])

    @property
    def done(self):
        # implement the custom done property
        done = False
        root_pos = self.sim.data.get_site_xpos(self.root).copy()
        # print("root position: ", root_pos)
        if root_pos[2] < 0.8:
            done = True
            print("done")
        return done

    @property
    def reward(self):
        # implement the reward
        alive_bonus = 5.0
        root_pos = self.sim.data.get_site_xpos(self.root).copy()
        lin_vel_cost = 1.0e+3 * (root_pos[2] - self.pos_before[2])
        quad_ctrl_cost = 0.1e-3 * np.square(self.sim.data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(self.sim.data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        return reward

    def step(self, a):
        self.pos_before = self.sim.data.get_site_xpos(self.root).copy()
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
