import os
import pickle
import logging
from abc import abstractmethod
from marshmallow_dataclass import dataclass
from typing import Type, Union, Dict, Optional, Tuple

import numpy as np
import gym
import torch

from robot_policy.rl.common import Agent, AgentConfig
from robot_policy.rl.common.base_buffer import ReplayBuffer
from robot_utils.py.utils import load_dataclass


@dataclass
class OffPolicyConfig(AgentConfig):
    buffer_capacity:        int = 1e5
    gamma:                  float = 0.99        # the discount factor
    gae_lambda:             float = 0.95        #
    use_sde:                bool = False        # use state dependent estimation
    sde_sample_freq:        int = -1
    num_env:                int = 1             # number of environments
    bootstrap_steps:        int = 100           # simulate "bootstrap_steps" to collect data for warm start (train dynamics)
    dyn_train_freq:         int = 50            # train the dynamics model in every "dyn_retrain_freq" steps
    dyn_train_mini_epochs:  int = 10            # mini epochs to train dynamics model


class OffPolicyAgent(Agent):
    def __init__(self, config: Optional[Dict], model_path: str):
        super(OffPolicyAgent, self).__init__(config, model_path)

        self.buffer_file = os.path.join(self.model_path, "buffer.pkl")
        self.cumulative_steps = 0
        self.progress = 0.0

    @abstractmethod
    def _load_config(self, config: Union[str, Dict, None]) -> None:
        self.c = load_dataclass(OffPolicyConfig, config) if config else torch.load(self.agent_config_file)

    def _init_agent(self, config: Dict):
        super(OffPolicyAgent, self)._init_agent(config)
        self.buffer = ReplayBuffer(
            capacity=self.c.buffer_capacity,
            a_dim=self.a_dim,
            s_dim=self.s_dim,
            device=self.device,
            n_envs=self.c.num_env,
        )

    def _get_action(self) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():   # unscaled, in [lower, upper]
            if self.cumulative_steps <= self.c.bootstrap_steps:
                act = self.env.action_space.sample()
            else:
                act, _ = self.predict(self._last_obs)

        scaled_act = self.policy.scale_action(act)  # in range [-1, 1]
        return act, scaled_act

    def save_buffer(self) -> None:
        with open(self.buffer_file, 'w') as buffer_file:
            pickle.dump(self.buffer, buffer_file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_buffer(self) -> None:
        with open(self.buffer_file, "r") as buffer_file:
            self.buffer = pickle.load(buffer_file)
            assert isinstance(self.buffer, ReplayBuffer)

    def rollout(self, render: bool = False):
        assert self._last_obs is not None, "No previous observation was provided"

        for n_steps in range(self.c.rollout_steps):
            with torch.no_grad():
                act, scaled_act = self._get_action()

            next_obs, rew, done, infos = self.env.step(act)
            if render:
                self.env.render()
            if done:
                self.env.reset()

            self.cumulative_steps += self.c.num_env
            self.progress_bar.update()

            if isinstance(self.env.action_space, gym.spaces.Discrete):
                scaled_act = scaled_act.reshape(-1, 1)

            self.buffer.add(self._last_obs, scaled_act, rew, next_obs, done, infos)
            self._last_obs = next_obs
            self._last_done = done

        # with torch.no_grad():
        #     # compute the last value
        #     obs_tensor = torch.as_tensor(next_obs).to(self.device)
        #     _, value, _ = self.policy.forward(obs_tensor)
        #
        # self.buffer.compute_returns_and_advantage(last_values=value, done=done)
        return True

    @abstractmethod
    def train(self):
        raise NotImplementedError

    def learn(self):
        self._setup_learn()
        n_rollouts = 0
        render = False
        while self.cumulative_steps < self.c.max_steps:
            done = self.rollout(render)
            self.progress = float(self.cumulative_steps) / float(self.c.max_steps)
            if not done:
                logging.warning("stop training, rollout failed")
                break
            self.train()
            n_rollouts += 1
            render = n_rollouts % self.c.render_every_n_rollout == 0
            if n_rollouts % self.c.play_every_n_rollout == 0:
                self.play()
            self.save_model()
        logging.info('finished learning')



