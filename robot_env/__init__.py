import gym
#from gym.envs import classic_control, box2d, mujoco
from gym.envs import classic_control
from pybullet_envs import gym_pendulum_envs, gym_manipulator_envs, gym_locomotion_envs
# from pybullet_envs.deep_mimic.gym_env.deep_mimic_env import HumanoidDeepBulletEnv, HumanoidDeepMimicBackflipBulletEnv
from pybullet_envs.bullet.racecarGymEnv import RacecarGymEnv
from pybullet_envs.bullet.minitaur_gym_env import MinitaurBulletEnv
from robot_env.utils import NormalizeActionEnv

from robot_env.gym_env import GymPendulumEnv
#from robot_env.mjc_env.armar4 import Armar4Env, Armar4StandingEnv

# https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs
# https://github.com/openai/gym/tree/master/gym/envs


env_list = {
    "InvertedDoublePendulumBulletEnv":  gym_pendulum_envs.InvertedDoublePendulumBulletEnv,
    "InvertedPendulumSwingUpBulletEnv": gym_pendulum_envs.InvertedPendulumSwingupBulletEnv,
    'InvertedPendulumBulletEnv':        gym_pendulum_envs.InvertedPendulumSwingupBulletEnv,
    #'HalfCheetah':                      mujoco.HalfCheetahEnv,
    'HalfCheetahBulletEnv':             gym_locomotion_envs.HalfCheetahBulletEnv,
    'HopperBulletEnv':                  gym_locomotion_envs.HopperBulletEnv,
    'AntBulletEnv':                     gym_locomotion_envs.AntBulletEnv,
    'ReacherBulletEnv':                 gym_manipulator_envs.ReacherBulletEnv,
    'PendulumEnv':                      classic_control.PendulumEnv,
    'CartPoleEnv':                      classic_control.CartPoleEnv,
    'Walker2DEnv':                      gym_locomotion_envs.Walker2DBulletEnv,
    #'BipedalWalker':                    box2d.BipedalWalker,
    #'LunarLanderContinuous':            box2d.LunarLanderContinuous,
    # 'DeepMimic':                        HumanoidDeepBulletEnv,
    # 'DeepMimicBackFlip':                HumanoidDeepMimicBackflipBulletEnv,
    'Minitaur':                         MinitaurBulletEnv,
    'RaceCarGymEnv':                    RacecarGymEnv,
    # Note: the following are the custome environments
    'GymPendulumEnv':                   GymPendulumEnv,
    #'Armar4Env':                        Armar4Env,
    #'Armar4StandingEnv':                Armar4StandingEnv
}


def get_env_list():
    out_str = ''
    for env_name in env_list.keys():
        out_str += env_name + '\n'
    return out_str


def get_env_from_name(env_name: str, render: bool = False, action_wrapper=False, make=False):
    if make:
        return gym.make(env_name)
    try:
        if action_wrapper:
            env = NormalizeActionEnv(env_list[env_name](render=render))
        else:
            env = env_list[env_name](render=render)
    except TypeError as err:
        if action_wrapper:
            env = NormalizeActionEnv(env_list[env_name]())
        else:
            env = env_list[env_name]()
    return env
