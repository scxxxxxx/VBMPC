import os
import click
import numpy as np
import gym
from gym import wrappers

from marshmallow_dataclass import dataclass
from typing import Union, Dict

from robot_env import get_env_from_name

from robot_policy.mpc.mpc_common import mpc_factory, ReplayBuffer
from robot_policy.mpc.mpc_common.twin_env import DynamicsModel1
from robot_policy.mpc.mpc_common.dynamic_optimizer import DynamicsModelOptimizer

from robot_utils.torch.torch_utils import get_device
from robot_utils.py.utils import create_path, load_dataclass, load_dict_from_yaml
from robot_utils.gym.gym_utils import get_space_dim
from robot_utils.torch.torch_utils import set_random_seeds, init_torch
device = init_torch()


@dataclass
class MPCConfig:
    env_name:           str
    env_init_state:     list = None
    use_gpu:            bool = True
    frame_skip:         int = 5
    num_rollouts:       int = 100
    horizon:            int = 15
    lamdba:             float = 1.0
    a_init:             list = None
    noise_sigma:        list = None
    max_epoch_steps:    int = 200
    max_frames:         int = 10000
    epsilon:            float = 1.0
    model_lr:           float = 3e-4
    policy_lr:          float = 3e-4
    seed:               int = 12345
    model_iter:         int = 100
    bootstrap_steps:    int = 0
    retrain_steps:      int = 50
    model_lambda:       float = 0.95
    method:             str = 'mppi'
    model:              str = 'mlp'
    render:             bool = False
    reward_type:        str = 'mlp'
    replay_buffer_size: int = 100000
    log:                bool = True
    done_util:          bool = True

    def __post_init__(self):
        self.device = get_device(self.use_gpu)


class MPCTrainer:
    """
    The MPC trainer takes care of training the dynamics model, the value function and the actor model (MPC algorithm)
    """
    def __init__(self, config_file: Union[str, Dict, None] = None, model_path: str = None):
        self.c = load_dataclass(MPCConfig, config_file['mpc'])
        default_path_name = os.path.join('/saved_models/', f"{self.c.env_name}_{self.c.method}_{self.c.reward_type}")
        self.model_path = model_path if model_path else default_path_name
        create_path(self.model_path)

        set_random_seeds(24, use_gpu=True)
        self._setup_env()
        self._setup_buffer()
        self._setup_dynamics(config_file['model_value'])
        self._setup_mpc_actor()

    def _setup_env(self):
        self.env = get_env_from_name(self.c.env_name, self.c.render)
        self.env = wrappers.Monitor(self.env, '/tmp/mppi/', force=True)
        self.env.reset()
        if self.c.env_init_state:
            self.env.env.state = self.c.env_init_state

        if isinstance(self.env.action_space, gym.spaces.Box):
            self.a_max = self.env.action_space.high
            self.a_min = self.env.action_space.low

        self.a_dim = get_space_dim(self.env.action_space)
        self.s_dim = get_space_dim(self.env.observation_space)

    def _setup_buffer(self):
        self.model_replay_buffer = ReplayBuffer(self.c.replay_buffer_size)

    def _setup_dynamics(self, config):
        self.dynamics = DynamicsModel1(self.s_dim, self.a_dim, config, self.c.device, self.c.env_name).to(self.c.device)
        self.optimizer = DynamicsModelOptimizer(self.dynamics,
                                                self.model_replay_buffer,
                                                lr=self.c.model_lr,
                                                lam=self.c.model_lambda,
                                                max_steps=self.c.max_frames)

    def _setup_mpc_actor(self):
        self.mpc_actor = mpc_factory(self.c.method)(
            dynamics=self.dynamics,
            running_reward=self.env.reward,
            noise_sigma=self.c.noise_sigma,
            num_rollouts=self.c.num_rollouts,
            horizon=self.c.horizon,
            lambda_=self.c.lamdba,
            a_min=self.a_min,
            a_max=self.a_max,
            a_init=self.c.a_init,
            device=self.c.device
        )

    def train(self):
        total_reward = 0
        self.env.env.state = self.c.env_init_state
        if self.c.bootstrap_steps > 0:
            action = np.random.uniform(low=self.a_min, high=self.a_max)
        else:
            action = self.mpc_actor.update_action(state)
        state = self.env.state.copy()
        for frame_idx in range(1, self.c.max_frames+1):
            next_state, reward, done, _   = self.env.step(action)
            if frame_idx <= self.c.bootstrap_steps:
                next_action = np.random.uniform(low=self.a_min, high=self.a_max)
            else:
                next_action = self.mpc_actor.update_action(next_state)
                total_reward += reward

            self.model_replay_buffer.push(state, action, reward, next_state, next_action, done)
            if self.c.render:
                self.env.render()

            di = frame_idx % self.c.retrain_steps
            if frame_idx == self.c.bootstrap_steps or frame_idx > self.c.bootstrap_steps and di == 0:
                self.optimizer.bootstrap_model(bootstrap_steps=100)

            state = next_state
            action = next_action

        print("Total reward %f", total_reward)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--config",   "-c",   default='',      help="the configuration file name")
@click.option("--path",     "-p",   default='',      help="the working directory for this experiments")
def main(config, path):
    config = "../mpc_train/config/mppi_model_fnn_reward_fnn.yaml"
    path = "trained_model/"
    config = load_dict_from_yaml(config)
    mppi = MPCTrainer(config_file=config, model_path=path)
    mppi.train()


if __name__ == "__main__":
    main()
