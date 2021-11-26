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
    # experiment settings
    env_name:           str
    use_gpu:            bool = True
    env_init_state:     list = None
    frame_skip:         int = 5
    render:             bool = False
    seed:               int = 12345

    # MPC algorithm
    mpc_method:         str = 'mppi'
    num_rollouts:       int = 100
    horizon:            int = 15
    lamdba:             float = 1.0
    a_init:             list = None
    noise_sigma:        list = None

    # Settings for training dynamics model
    max_frames:         int = 10000     # maximum simulated frames
    max_epoch_steps:    int = 200       # maximum steps in each epoch of the simulation
    bootstrap_steps:    int = 100       # maximum steps used for bootstrap the dynamics training
    retrain_steps:      int = 50        # frequency of training the dynamic model
    model_lr:           float = 3e-4    # learning rate of dynamic model
    model_iter:         int = 100       # number iteration for each training the dynamics model
    batch_size:         int = 250       # size of sampled data from the replay buffer
    replay_buffer_size: int = 100000    # maximum size of replay buffer
    l1_norm:            bool = False    # enalbe l1 norm loss (used in EQL models)
    prune:              bool = False    # enable weights pruning (used in EQL moddels)

    def __post_init__(self):
        self.device = get_device(self.use_gpu)


class MPCTrainer:
    """
    The MPC trainer takes care of training the dynamics model, the value function and the actor model (MPC algorithm)
    """
    def __init__(self, config_file: Union[str, Dict, None] = None, model_path: str = None):
        self.c = load_dataclass(MPCConfig, config_file['mpc'])
        default_path_name = os.path.join('./saved_models/',
                                         f"{self.c.env_name}_{self.c.mpc_method}_{config_file['dynamics']['type']}")
        self.model_path = model_path if model_path else default_path_name
        create_path(self.model_path)

        set_random_seeds(24, use_gpu=True)
        self._setup_env()
        self._setup_buffer()
        self._setup_dynamics(config_file['dynamics'])
        self._setup_mpc_actor()

        self.first_run = True

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
        self.optimizer = DynamicsModelOptimizer(
            dynamics=self.dynamics,
            replay_buffer=self.model_replay_buffer,
            lr=self.c.model_lr,
            max_steps=self.c.max_frames,
            bootstrap_steps=self.c.bootstrap_steps,
            l1_norm=self.c.l1_norm,
            prune=self.c.prune
        )

    def _setup_mpc_actor(self):
        self.mpc_actor = mpc_factory(self.c.mpc_method)(
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

    def _get_action(self, state, frame_idx):
        if self.first_run:
            self.first_run = False
            return np.random.uniform(low=self.a_min, high=self.a_max)
        if frame_idx <= self.c.bootstrap_steps:
            return np.random.uniform(low=self.a_min, high=self.a_max)
        else:
            return self.mpc_actor.update_action(state)


    def train(self):
        frame_idx = 1
        rewards = []

        ep_num = 1
        self.mpc_actor.reset()
        while frame_idx < self.c.max_frames:
            state = self.env.reset()
            action = self._get_action(state, frame_idx)

            episode_reward = 0
            for step in range(1, self.c.max_epoch_steps + 1):
                next_state, reward, done, _ = self.env.step(action.copy())
                next_action = self._get_action(next_state, frame_idx)
                self.model_replay_buffer.push(state, action, reward, next_state, next_action, done)

                # if frame_idx == self.c.bootstrap_steps:
                #     print("--" * 20 + " bootstrap dynamics " + "--" * 20)
                #     self.optimizer.bootstrap_model(self.c.bootstrap_steps)
                if frame_idx == self.c.bootstrap_steps or \
                        frame_idx > self.c.bootstrap_steps and frame_idx % self.c.retrain_steps == 0:
                    self.optimizer.update_model(frame_idx, self.c.batch_size, mini_iter=self.c.model_iter)

                state = next_state
                action = next_action
                episode_reward += reward[0, 0]
                frame_idx += 1

                if (not self.c.render and frame_idx > 0.8 * self.c.max_frames) or self.c.render:
                    self.env.render()

                if done:
                    break
            else:
                self.env.stats_recorder.save_complete()
                self.env.stats_recorder.done = True

            print(f'episode: {ep_num:>4} \t reward: {episode_reward:>14.8f}')
            rewards.append([frame_idx, episode_reward])
            ep_num += 1


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--config",   "-c",   default='',      help="the configuration file name")
@click.option("--path",     "-p",   default='',      help="the working directory for this experiments")
def main(config, path):
    config = load_dict_from_yaml(config)
    mppi = MPCTrainer(config_file=config, model_path=path)
    mppi.train()


if __name__ == "__main__":
    main()
