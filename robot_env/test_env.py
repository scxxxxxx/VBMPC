import logging
from robot_env import get_env_from_name
from robot_env.utils import get_robot_path
from robot_utils.gym.gym_utils import get_space_dim
from robot_utils.torch.torch_utils import init_torch

# env = gym.make("LunarLander-v2") # RocketLander-v0 | LunarLander-v2 | MountainCar-v0 | CartPole-v0
# env = get_env_from_name("LunarLanderContinuous")
env_name = "Armar6BimanualTable-v0"
logging.info(f"{env_name}: {get_robot_path()}")
env = get_env_from_name(env_name=env_name, render=True, action_wrapper=False, make=True)
env.reset()
step = 0

a_dim = get_space_dim(env.action_space)
s_dim = get_space_dim(env.observation_space)
print('State size: {}'.format(s_dim))
print('Action size: {}'.format(a_dim))

PRINT_DEBUG_MSG = False


while True:
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    step += 1

    if PRINT_DEBUG_MSG:
        print("Step          ", step)
        print("Action Taken  ", action)
        print("Observation   ", observation)
        print("Reward Gained ", reward)
        print("Info          ", info, end='\n\n')

    if done:
        print("Simulation done.")
        break
env.close()