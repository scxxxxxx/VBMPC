import gym
import numpy as np


class NormalizeActionEnv(gym.ActionWrapper):
    def action(self, action):
        if isinstance(self.action_space, gym.spaces.Box):
            low_bound = self.action_space.low
            upper_bound = self.action_space.high

            # action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
            action = np.clip(action, low_bound, upper_bound)

        # ic(action)
        return action

    def reverse_action(self, action):
        if isinstance(self.action_space, gym.spaces.Box):
            low_bound = self.action_space.low
            upper_bound = self.action_space.high

            # action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
            action = np.clip(action, low_bound, upper_bound)

        return action


# TODO add normalized state env
class NormalizeObsEnv(gym.ObservationWrapper):
    def observation(self, observation):
        raise NotImplementedError


class NormalizedEnvWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(self.action(action))
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        raise NotImplementedError

    def action(self, action):
        raise NotImplementedError

    def reverse_action(self, action):
        raise NotImplementedError
