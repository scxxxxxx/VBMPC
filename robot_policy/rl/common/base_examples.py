from typing import Optional, Dict, Tuple

import gym
import torch
from marshmallow_dataclass import dataclass

from robot_policy.rl.common.base_policy import PolicyConfig, PolicyFactory, BasePolicy
from robot_utils.py.utils import load_dataclass


@dataclass
class ExamplePolicyConfig(PolicyConfig):
    pass


@PolicyFactory.register("ExamplePolicy")
class ExamplePolicy(BasePolicy):
    def __init__(self, env: gym.Env, config: Optional[Dict], model_path: str, device: torch.device):
        super(ExamplePolicy, self).__init__(env, config, model_path, device)

    def _load_config(self, config: Dict):
        """Implement your own method to load configuration"""
        self.c = load_dataclass(ExamplePolicyConfig, config) if config else torch.load(self.policy_config_file)

    def _init_policy_networks(self) -> None:
        """
        Create the networks for shared_net, action_net and value_net
        """
        pass

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic), return the action, values, and the log_prob(action)
        """
        pass

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Given observation, get the next (default: deterministic) action.
        """
        pass

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given observation, get the values, log probability and entropy of the action distribution.
        """
        pass



