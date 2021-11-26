from gym.envs.registration import registry, register, make, spec
from robot_env.gym_env.classic_control import GymPendulumEnv
#from robot_env.gym_env.box2d import RocketLander

register(
    id='GymPendulumEnv-v1',
    entry_point='robot_env.gym_env.classic_control:GymPendulumEnv',
    max_episode_steps=50000,
)

# register(
#     id='RocketLander-v1',
#     entry_point='robot_env.gym_env.box2d:RocketLander',
#     max_episode_steps=1000,
#     reward_threshold=0,
# )