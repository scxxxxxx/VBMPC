import os
import click
import pickle
import logging
import numpy as np
import torch

from icecream import ic
from marshmallow_dataclass import dataclass
from typing import Union, Dict

from robot_env import get_env_from_name
from robot_policy.mpc.mpc_common import mpc_factory, ReplayBuffer
from robot_policy.mpc.mpc_common.twin_env import DynamicsModel
from robot_policy.mpc.mpc_common.dynamic_optimizer import DynamicsModelOptimizer

from robot_utils.torch.torch_utils import get_device
from robot_utils.py.utils import load_dataclass, load_dict_from_yaml
from robot_utils.gym.gym_utils import get_space_dim
from robot_utils.py.utils import create_path
from robot_utils.torch.torch_utils import init_torch
device = init_torch()


@dataclass
class MPCConfig:
    env_name:           str
    use_gpu:            bool = True
    frame_skip:         int = 5
    max_steps:          int = 200
    max_frames:         int = 10000
    epsilon:            float = 1.0
    model_lr:           float = 3e-4
    policy_lr:          float = 3e-4
    seed:               int = 12345
    horizon:            int = 5
    model_iter:         int = 2
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
    def __init__(self, config_file: Union[str, Dict, None] = None, model_path: str = None):
        self.c = load_dataclass(MPCConfig, config_file['mpc'])
        default_path_name = os.path.join('./saved_models/', f"{self.c.env_name}_{self.c.method}_{self.c.reward_type}")
        self.model_path = model_path if model_path else default_path_name
        create_path(self.model_path)

        self._setup_env()
        self._setup_buffer()
        self._setup_model(config_file['model_value'])
        self._setup_mpc_actor()

    def _setup_env(self):
        self.env = get_env_from_name(self.c.env_name, self.c.render)
        self.env.reset()

        self.a_dim = get_space_dim(self.env.action_space)
        self.s_dim = get_space_dim(self.env.observation_space)
        ic(self.a_dim, self.s_dim)

    def _setup_buffer(self):
        self.model_replay_buffer = ReplayBuffer(self.c.replay_buffer_size)

    def _setup_model(self, config):
        self.model = DynamicsModel(self.s_dim, self.a_dim, config, self.c.device, self.c.env_name).to(self.c.device)
        self.optimizer = DynamicsModelOptimizer(self.model, self.model_replay_buffer, lr=self.c.model_lr,
                                                lam=self.c.model_lambda, max_steps=self.c.max_frames)

    def _setup_mpc_actor(self):
        mpc_method = mpc_factory(self.c.method)
        self.mpc_planner = mpc_method(self.model, horizon=self.c.horizon, eps=self.c.epsilon, device=self.c.device)

    def train(self):
        max_frames = self.c.max_frames
        frame_skip = self.c.frame_skip
        max_steps = 1000

        frame_idx = 0
        rewards = []
        batch_size = 256

        ep_num = 0
        while frame_idx < max_frames:
            state = self.env.reset()
            self.mpc_planner.reset()

            action = self.mpc_planner.update(state)

            episode_reward = 0
            done = False
            for step in range(max_steps):
                for _ in range(frame_skip):
                    next_state, reward, done, _ = self.env.step(action.copy())

                next_action = self.mpc_planner.update(next_state)

                if self.c.method == 'ilqr' or self.c.method == 'shooting':
                    eps = 1.0 * (0.995 ** frame_idx)
                    next_action = next_action + np.random.normal(0., eps, size=(self.a_dim,))

                self.model_replay_buffer.push(state, action, reward, next_state, next_action, done)

                if len(self.model_replay_buffer) > batch_size:
                    self.optimizer.update_model(frame_idx, batch_size, mini_iter=self.c.model_iter)

                state = next_state
                action = next_action
                episode_reward += reward
                frame_idx += 1

                if (not self.c.render and frame_idx > 0.8 * max_frames) or self.c.render:
                    self.env.render()

                if frame_idx % 100 == 0:
                    last_reward = rewards[-1][1] if len(rewards) > 0 else 0
                    logging.info('frame : {}/{}: \t last rew: {}'.format(frame_idx, max_frames, last_reward))

                    if self.c.log:
                        pickle.dump(rewards, open(self.model_path + 'reward_data' + '.pkl', 'wb'))
                        torch.save(self.model.state_dict(), self.model_path + 'model_' + str(frame_idx) + '.pt')

                if self.c.done_util:
                    if done:
                        break

            logging.info('episodic rew: {} \t {}'.format(ep_num, episode_reward))
            rewards.append([frame_idx, episode_reward])
            ep_num += 1

        if self.c.log:
            pickle.dump(rewards, open(self.model_path + 'reward_data' + '.pkl', 'wb'))
            torch.save(self.model.state_dict(), self.model_path + 'model_' + 'final' + '.pt')


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--config",   "-c",   default='',      help="the configuration file name")
@click.option("--path",     "-p",   default='',      help="the working directory for this experiments")
def main(config, path):
    config = load_dict_from_yaml(config)
    mppi = MPCTrainer(config_file=config, model_path=path)
    mppi.train()


if __name__ == '__main__':
    main()
