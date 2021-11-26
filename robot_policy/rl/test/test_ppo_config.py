import os
import click
import logging
import gym
import torch
from icecream import ic, install
from tqdm import trange
from tabulate import tabulate
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Type, Union, Dict
from marshmallow_dataclass import dataclass

from robot_utils.torch.torch_utils import get_device, set_random_seeds, init_torch
from robot_utils.py.utils import load_dataclass_from_dict, load_dict_from_yaml, load_dataclass, create_path
from robot_utils.gym.gym_utils import get_space_dims, get_action_bound
from robot_env import get_env_from_name
init_torch()
install()


@dataclass
class AgentConfig:
    # setup
    env_name:       Union[str, None] = None
    seed:           Union[int, None] = None
    use_gpu:        bool = True

    # train
    learning_rate:  float = 1e-4
    rollout_steps:  int = 1000
    max_steps:      int = 10000
    batch_size:     int = 64
    epochs:         int = 1000
    max_grad_norm:  float = 0.5

    def __post_init__(self):
        self.device = get_device(self.use_gpu)
        self.env = get_env_from_name(env_name=self.env_name, render=True, action_wrapper=False, make=True)
        self.a_dim, self.s_dim = get_space_dims(self.env)
        self.a_min, self.a_max = get_action_bound(self.env)
        if self.seed is not None:
            set_random_seeds(self.seed)

@dataclass
class OnPolicyConfig(AgentConfig):
    gamma:              float = 0.99
    gae_lambda:         float = 0.95
    ent_coef:           float = 0.0
    val_coef:           float = 0.5
    use_sde:            bool = False
    sde_sample_freq:    int = -1
    num_env:            int = 1


@dataclass
class PPOConfig(OnPolicyConfig):
    clip_range:     float = 0.2
    clip_range_vf:  Union[float, None] = None
    target_kl:      Union[float, None] = None


class Agent(ABC):
    def __init__(self,
                 # config_type:   Union[AgentConfig, None] = AgentConfig,
                 config:        Union[Dict, None] = None,
                 model_path:    str = None):
        # Configuration
        self.model_path = create_path(model_path if model_path else "./saved_model/base_agent")
        self.agent_config_file = os.path.join(self.model_path, f"{self.__class__.__name__}_actor_config.pt")
        if config:
            self._load_config(config['agent'])
        else:
            self.c = torch.load(self.agent_config_file)

        self.device = self.c.device
        self.env = self.c.env

    @abstractmethod
    def _load_config(self, config: Dict):
        self.c = load_dataclass_from_dict(AgentConfig, config)

    def print(self, param=True, info=True, *args, **kwargs):
        if param:
            print("=" * 45, " Printing Parameters of {} ".format(self.__class__.__name__), "=" * 45)
            for name, param in self.policy.named_parameters():
                if param.requires_grad:
                    ic(name, param.data)
                print("-" * 120)
        if info:
            print("=" * 45, "  Printing Info Dict of {} ".format(self.__class__.__name__), "=" * 45)
            print(tabulate([(k, v) for k, v in self.c.__dict__.items()], headers=["name", "value"], tablefmt="github"))
            print("-" * 120)


class OnPolicyAgent(Agent):
    def __init__(self,
                 config: Union[Dict, None] = None,
                 model_path: str = None):
        super(OnPolicyAgent, self).__init__(config, model_path)

        self.cumulative_steps = 0
        self.progress = 0.0

    @abstractmethod
    def _load_config(self, config: Dict):
        self.c = load_dataclass_from_dict(OnPolicyConfig, config)


class PPO(OnPolicyAgent):
    def __init__(self,
                 config: Union[Dict, None] = None,
                 model_path: str = None):
        super(PPO, self).__init__(config, model_path)

        self.clip_range = self.c.clip_range
        self.clip_range_vf = self.c.clip_range_vf
        self.target_kl = self.c.target_kl

    def _load_config(self, config):
        self.c = load_dataclass_from_dict(PPOConfig, config)



config_file = "./config/ppo.yaml"
config = load_dict_from_yaml(config_file)
# ic(config["agent"])
# c = load_dataclass(AgentConfig, config["agent"])
# policies[policy_type](env, config=config, model_path=model_path)
# ic(c)
ppo = PPO(config=config, model_path="./saved_model/test_config")
ppo.print(param=False, info=True)
