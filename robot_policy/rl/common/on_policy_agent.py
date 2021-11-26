import logging
from abc import abstractmethod
from marshmallow_dataclass import dataclass
from typing import Type, Union, Dict, Optional

import numpy as np
import gym
import torch

from robot_policy.rl.common import Agent, AgentConfig
from robot_policy.rl.common.base_buffer import RolloutBuffer
from robot_utils.py.utils import load_dataclass


@dataclass
class OnPolicyConfig(AgentConfig):
    gamma:              float = 0.99
    gae_lambda:         float = 0.95
    ent_coef:           float = 0.0
    val_coef:           float = 0.5
    use_sde:            bool = False
    sde_sample_freq:    int = -1
    num_env:            int = 1


class OnPolicyAgent(Agent):
    def __init__(self, config: Optional[Dict], model_path: str):
        super(OnPolicyAgent, self).__init__(config, model_path)

        self.cumulative_steps = 0
        self.progress = 0.0

    @abstractmethod
    def _load_config(self, config: Union[str, Dict, None]) -> None:
        self.c = load_dataclass(OnPolicyConfig, config) if config else torch.load(self.agent_config_file)

    def _init_agent(self, config: Dict):
        super(OnPolicyAgent, self)._init_agent(config)
        a_dim = 1 if isinstance(self.env.action_space, gym.spaces.Discrete) else self.a_dim
        self.buffer = RolloutBuffer(
            rollout_steps=self.c.rollout_steps,
            a_dim=a_dim,
            s_dim=self.s_dim,
            device=self.device,
            gamma=self.c.gamma,
            gae_lambda=self.c.gae_lambda,
            n_envs=self.c.num_env,
        )

    def _get_action(self, obs: torch.Tensor):
        with torch.no_grad():
            act, val, logp = self.policy(obs)

        act = self.policy.unscale_action(act.cpu().numpy())
        return act, val, logp

    def rollout(self, render: bool = False):
        """ roll out in the environment to collect data """
        assert self._last_obs is not None, "No previous observation was provided"
        self.buffer.reset()

        for n_steps in range(self.c.rollout_steps):
            with torch.no_grad():
                obs = torch.as_tensor(self._last_obs).float().to(self.device)
                act, val, logp = self._get_action(obs)
                scaled_act = self.policy.scale_action(act)

            next_obs, rew, done, infos = self.env.step(act)
            if render:
                self.env.render()
            if done:
                self.env.reset()

            self.cumulative_steps += self.c.num_env
            self.progress_bar.update()
            # self.cumulative_steps += self.env.num_envs

            if isinstance(self.env.action_space, gym.spaces.Discrete):
                scaled_act = scaled_act.reshape(-1, 1)  # Reshape in case of discrete action

            self.buffer.add(self._last_obs, scaled_act, rew, self._last_done, val, logp)
            self._last_obs = next_obs
            self._last_done = done

        with torch.no_grad():
            # compute the last value
            obs_tensor = torch.as_tensor(next_obs).to(self.device)
            _, value, _ = self.policy.forward(obs_tensor)

        self.buffer.compute_returns_and_advantage(last_values=value, done=done)
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


