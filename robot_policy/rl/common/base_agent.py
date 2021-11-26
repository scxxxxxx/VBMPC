import os
import time
import logging
import numpy as np
import torch
import gym
import shutil

from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Type, Union, Dict, Any
from marshmallow_dataclass import dataclass
from tabulate import tabulate

from robot_utils.torch.torch_utils import set_random_seeds, get_device
from robot_utils.gym.gym_utils import get_space_dims, get_action_bound
from robot_utils.py.utils import create_path, load_dataclass, save_to_yaml, load_dict_from_yaml

from robot_policy.rl.common.base_policy import PolicyFactory
from robot_env import get_env_from_name


@dataclass
class AgentConfig:
    # setup
    env_name:               Union[str, None] = None
    env_config:             Optional[Dict[str, Any]] = None
    seed:                   Union[int, None] = None
    use_gpu:                bool = True
    render_every_n_rollout: int = 100
    play_every_n_rollout:   int = 100

    policy_type:            str = None

    # train
    rollout_steps:          int = 200
    max_steps:              int = 10000
    batch_size:             int = 64
    train_mini_epochs:      int = 1000
    max_grad_norm:          float = 0.5
    buffer_size:            int = 100000  # maximum size of replay buffer


class Agent(ABC):
    def __init__(self, config: Union[str, Dict, None] = None, model_path: str = None):
        # Note: initialize agent configurations
        self.model_path = create_path(f"./saved_model/{self.name}" if model_path is None else model_path)
        self.agent_config_file = os.path.join(self.model_path, f"{self.name}_agent_config.yaml")
        if config is None:
            config = load_dict_from_yaml(self.agent_config_file)
        elif isinstance(config, str):
            shutil.copy(config, self.agent_config_file)
            config = load_dict_from_yaml(config)
        else:
            save_to_yaml(config, self.agent_config_file)
        self._load_config(config['agent'])

        # Note: setup environment
        self.device = get_device(self.c.use_gpu)
        self._init_env()
        self.a_dim, self.s_dim = get_space_dims(self.env)
        self.a_min, self.a_max = get_action_bound(self.env)

        if self.c.seed is not None:
            gym.logger.set_level(40)
            set_random_seeds(self.c.seed, self.c.use_gpu)

        # Note: initialize policy (and buffer)
        self._init_agent(config)

        # Note: logging
        self.log_bar = tqdm(total=0, position=1, bar_format='{desc}', leave=True)
        self.progress_bar = tqdm(total=self.c.max_steps, desc="iteration", leave=True)

    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def _load_config(self, config: Dict) -> None:
        self.c = load_dataclass(AgentConfig, config)

    def _init_env(self):
        self.env = get_env_from_name(env_name=self.c.env_name, **self.c.env_config)

    @abstractmethod
    def _init_agent(self, config: Dict):
        self._init_policy(config)

    def _init_policy(self, config: Dict, **kwargs) -> None:
        """
        Initialize the policy networks with the given configuration dictionary.
        """
        policy_config = config['policy']
        policy_config.update({
            'a_dim': self.a_dim,
            's_dim': self.s_dim,
            'a_min': self.a_min,
            'a_max': self.a_max
        })
        self.policy = PolicyFactory.create_model(self.c.policy_type)(
            self.env, policy_config, self.model_path, self.device, **kwargs
        ).to(self.device)

    def _setup_learn(self) -> None:
        self.start_time = time.time()
        self._last_obs = self.env.reset()
        self._last_done = np.zeros((1,), dtype=np.bool)

    def optimize_step(self, loss, clipping_norm=None, retain_graph=False):
        """ optimization step """
        self.policy.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)

        if clipping_norm:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.c.max_grad_norm)
        self.policy.optimizer.step()

    def log_gradient_and_weight_information(self, network, optimizer):
        # log weight information
        total_norm = 0
        for name, param in network.named_parameters():
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        logging.info("Gradient Norm {}".format(total_norm))

        for g in optimizer.param_groups:
            learning_rate = g['lr']
            break
        logging.info("Learning Rate {}".format(learning_rate))

    @staticmethod
    def soft_update_network_param(current_model, target_model, tau):
        """
        updates the target model by taking a smaller step in the direction of current model, for a more stable training
        Args:
            current_model: the current model, whose parameters are continuously updated and used in exploration
            target_model: the target model, whose parameters will be updated
            tau: coefficient of the exponential low pass filter for update target model
        """
        for target_param, local_param in zip(target_model.parameters(), current_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def predict(self,
                obs: np.ndarray,
                state: Optional[np.ndarray] = None,
                mask: Optional[np.ndarray] = None,
                deterministic: bool = False
                ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get next unscaled action(s) in range [low, high]

        Args:
            obs: the input observation
            state: The last states (can be None, used in recurrent policies)
            mask: The last masks (can be None, used in recurrent policies)
            deterministic: deterministic or stochastic actions

        Returns: action and next state (for recurrent policy)

        """
        return self.policy.predict(obs, state, mask, deterministic)

    def save_model(self, best_param=None):
        self.policy.save_model()

    def load_model(self):
        self.policy.load_model()

    def print(self, param=True, info=True, *args, **kwargs) -> None:
        """
        Print the policy parameters and/or agent configurations in a table.
        """
        if param:
            print("=" * 45, " Printing Parameters of {} ".format(self.__class__.__name__), "=" * 45)
            for name, param in self.policy.named_parameters():
                if param.requires_grad:
                    print(name, param.data)
                print("-" * 120)
        if info:
            print("=" * 45, "  Configuration Dict of {} ".format(self.__class__.__name__), "=" * 45)
            print(tabulate([(k, v) for k, v in self.c.__dict__.items()], headers=["name", "value"], tablefmt="github"))
            print("-" * 120)

    def play(self, rollouts: int = 1, rollout_steps: int = 5000, progress: bool = False) -> None:
        """
        Execute the current policy for "rollouts" rollouts with "rollout_steps" per rollout.
        """
        if hasattr(self.env, "_max_episode_steps"):
            rollout_steps = min(rollout_steps, self.env._max_episode_steps)
        total_steps = rollouts * rollout_steps
        pbar = tqdm(total=total_steps, desc="iteration", leave=True) if progress else None
        for _ in range(rollouts):
            obs = self.env.reset()
            for _ in range(rollout_steps):
                action, _states = self.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                self.env.render()
                if progress:
                    pbar.update()
                if done:
                    break

    def start(self, rollouts: int = 1, rollout_steps: int = 5000, progress: bool = False) -> None:
        if hasattr(self.env, "_max_episode_steps"):
            rollout_steps = min(rollout_steps, self.env._max_episode_steps)
        for _ in range(rollouts):
            obs = self.env.reset()
            for _ in range(rollout_steps):
                action = self.policy.forward(obs)
                self.policy.update_action(obs)
                obs, reward, done, info = self.env.step(action)
                self.env.render()
                if done:
                    break
