import os
import numpy as np
import logging
from marshmallow_dataclass import dataclass
from typing import Union, Dict

from robot_policy.rl.common import Agent, AgentConfig
from robot_policy.rl.common.base_buffer import ReplayBuffer
from robot_policy.mpc.mpc_common.twin_env import TwinEnvFactory
from robot_policy.mpc.mpc_common.dynamic_optimizer import DynamicsModelOptimizer, DynamicsRewardModelOptimizer, CriticOptimizer
from robot_utils.py.utils import load_dataclass_from_dict


@dataclass
class IMPPIConfig(AgentConfig):
    # experiment settings
    env_init_state:     list = None
    frame_skip:         int = 5
    twin_env_type:      str = ""
    ac_type:           str = ""
    # render:             bool = False

    # MPC algorithm
    num_rollouts:       int = 100
    horizon:            int = 15
    lamdba:             float = 1.0
    a_init:             list = None
    noise_sigma:        list = None

    # Settings for training dynamics model
    bootstrap_steps:    int = 100       # maximum steps used for bootstrap the dynamics training
    retrain_steps:      int = 50        # frequency of training the dynamic model


class IMPPI(Agent):
    """
    The MPC trainer takes care of training the dynamics model, the value function and the actor model (MPC algorithm)
    """
    def __init__(self, config: Union[str, Dict, None], model_path: str):
        super(IMPPI, self).__init__(config, model_path)

        self.first_run = True
        self.cumulative_steps = 0

    def _load_config(self, config: Dict):
        self.c = load_dataclass_from_dict(IMPPIConfig, config)

    def _init_env(self):
        super(IMPPI, self)._init_env()
        # self.env = gym.wrappers.Monitor(self.env, '/tmp/mppi/', force=True)
        self.env.reset()
        if self.c.env_init_state:
            # self.env.env.state = self.c.env_init_state
            self.env.state = self.c.env_init_state

    def _init_agent(self, config: Dict):
        """ initialize buffer, dynamics model and MPPI policy """
        self.buffer = ReplayBuffer(self.c.buffer_size, a_dim=self.a_dim, s_dim=self.s_dim, device=self.device)
        # self._setup_dynamics(config)
        self._setup_twin_env(config)
        self._init_policy(config)

    def _setup_twin_env(self, config):
        # setup dynamics model
        twin_env_config = config['twin_env']
        twin_env_config["s_dim"] = self.s_dim
        twin_env_config["a_dim"] = self.a_dim
        assert self.c.twin_env_type != ""
        self.twin_env = TwinEnvFactory.create_model(self.c.twin_env_type, twin_env_config, self.model_path, self.device)
        # TODO find a way to unify the reward setting, from env, from twin_env
        # self.twin_env.reset_rew(self.env.reward)
        # setup dynamics model trainer
        dynamics_train_config = config['dynamics_train']
        dynamics_train_config['max_steps'] = self.c.max_steps
        dynamics_train_config['bootstrap_steps'] = self.c.bootstrap_steps
        self.optimizer = DynamicsRewardModelOptimizer(
            dynamics=self.twin_env,
            replay_buffer=self.buffer,
            **dynamics_train_config
        )

    # def _setup_dynamics(self, config):
    #     # setup dynamics model
    #     dynamics_model_config = config['dynamics_model']
    #     path = os.path.join(self.model_path, f"env_{self.c.env_name}_model_{dynamics_model_config['network_type']}")
    #     self.dynamics = DynamicsModel1(self.s_dim, self.a_dim, dynamics_model_config, self.device, path).to(self.device)
    #
    #     # setup dynamics model trainer
    #     dynamics_train_config = config['dynamics_train']
    #     dynamics_train_config['max_steps'] = self.c.max_steps
    #     dynamics_train_config['bootstrap_steps'] = self.c.bootstrap_steps
    #     self.optimizer = DynamicsModelOptimizer(
    #         dynamics=self.dynamics,
    #         replay_buffer=self.buffer,
    #         **dynamics_train_config
    #     )

    def _init_policy(self, config: Dict, **kwargs):
        super(IMPPI, self)._init_policy(
            config,
            twin_env=self.twin_env,
            terminal_cost=None
        )

    def _get_action(self, state):
        if self.first_run:
            self.first_run = False
            return np.random.uniform(low=self.a_min, high=self.a_max)
        if self.cumulative_steps <= self.c.bootstrap_steps:
            return np.random.uniform(low=self.a_min, high=self.a_max)
        else:
            return self.policy.forward(state)

    def learn(self):
        try:
            self.train()
        except KeyboardInterrupt:
            logging.warning("Interrupt training")
            # self.env.stats_recorder.save_complete()
            # self.env.stats_recorder.done = True

    def train(self):
        rewards = []

        n_rollouts = 1
        self.policy.reset()
        while self.cumulative_steps < self.c.max_steps:
            state = self.env.reset()
            action = self._get_action(state)
            render = n_rollouts % self.c.render_every_n_rollout == 0
            episode_reward = 0
            for step in range(1, self.c.rollout_steps + 1):
                next_state, reward, done, infos = self.env.step(action.copy().squeeze())
                next_action = self._get_action(next_state)
                self.buffer.add(state, action, reward, next_state.squeeze(), done, infos)

                if self.cumulative_steps == self.c.bootstrap_steps or \
                        self.cumulative_steps > self.c.bootstrap_steps and \
                        self.cumulative_steps % self.c.retrain_steps == 0:
                    self.optimizer.update_model(self.cumulative_steps, self.c.batch_size,
                                                mini_iter=self.c.train_mini_epochs)

                state = next_state
                action = next_action
                episode_reward += reward[0, 0]
                self.cumulative_steps += 1

                if render:
                    self.env.render()
                if done:
                    break
            # else:
            #     self.env.stats_recorder.save_complete()
            #     self.env.stats_recorder.done = True

            print(f'episode: {n_rollouts:>4} \t reward: {episode_reward:>14.8f}')
            rewards.append([self.cumulative_steps, episode_reward])
            n_rollouts += 1


# TODO try to add value function for the last rew in a rollout, the above might already work, see MPPI policy
class ActorCriticIMPPI(Agent):
    """
    The MPC trainer takes care of training the dynamics model, the value function and the actor model (MPC algorithm)
    """
    def __init__(self, config: Union[str, Dict, None], model_path: str):
        super(ActorCriticIMPPI, self).__init__(config, model_path)

        self.first_run = True
        self.cumulative_steps = 0

    def _load_config(self, config: Dict):
        self.c = load_dataclass_from_dict(IMPPIConfig, config)
        print(self.c)

    def _init_env(self):
        super(ActorCriticIMPPI, self)._init_env()
        # self.env = gym.wrappers.Monitor(self.env, '/tmp/mppi/', force=True)
        self.env.reset()
        if self.c.env_init_state:
            # self.env.env.state = self.c.env_init_state
            self.env.state = self.c.env_init_state

    def _init_agent(self, config: Dict):
        """ initialize buffer, dynamics model and MPPI policy """
        self.buffer = ReplayBuffer(self.c.buffer_size, a_dim=self.a_dim, s_dim=self.s_dim, device=self.device,
                                   sample_style="extended")
        self._setup_dynamics(config)
        self._init_policy(config)

    def _setup_dynamics(self, config: Dict) -> None:
        # setup dynamics model
        dynamics_model_config = config['dynamics_model']
        path = os.path.join(self.model_path, f"env_{self.c.env_name}_model_{dynamics_model_config['network_type']}")
        #self.dynamics = TwinEnvFactory.MLPDyn(self.s_dim, self.a_dim, dynamics_model_config, self.device, path).to(self.device)
        self.dynamics = TwinEnvFactory.create_model("MLPDyn", dynamics_model_config, path, self.device)

        # setup critic model
        critic_model_config = config['critic_model']
        path = os.path.join(self.model_path, f"env_{self.c.env_name}_value_{critic_model_config['network_type']}")
        self.critic = RewardModel(self.s_dim, self.a_dim, critic_model_config, self.device, path).to(self.device)

        # setup dynamics model trainer
        train_config = config['dyn_critic_train']
        train_config['max_steps'] = self.c.max_steps
        train_config['bootstrap_steps'] = self.c.bootstrap_steps

        self.optimizer = DynamicsCriticModelOptimizer(
            dynamics=self.dynamics,
            critic=self.critic,
            replay_buffer=self.buffer,
            **train_config,
        )


    def _init_policy(self, config: Dict, **kwargs):
        super(ActorCriticIMPPI, self)._init_policy(
            config,
            dynamics=self.dynamics,
            running_reward=self.critic,
            terminal_state_cost=None
        )

    def _get_action(self, state):
        if self.first_run:
            self.first_run = False
            return np.random.uniform(low=self.a_min, high=self.a_max)
        if self.cumulative_steps <= self.c.bootstrap_steps:
            return np.random.uniform(low=self.a_min, high=self.a_max)
        else:
            return self.policy.forward(state)

    def learn(self):
        try:
            self.train()
        except KeyboardInterrupt:
            logging.warning("Interrupt training")

    def train(self):
        rewards = []

        n_rollouts = 1
        self.policy.reset()
        while self.cumulative_steps < self.c.max_steps:
            state = self.env.reset()
            action = self._get_action(state)
            render = n_rollouts % self.c.render_every_n_rollout == 0
            episode_reward = 0
            for step in range(1, self.c.rollout_steps + 1):
                next_state, reward, done, infos = self.env.step(action.copy().squeeze())
                next_action = self._get_action(next_state)
                self.buffer.add(state, action, reward, next_state.squeeze(), done, infos)

                if self.cumulative_steps == self.c.bootstrap_steps or \
                        self.cumulative_steps > self.c.bootstrap_steps and \
                        self.cumulative_steps % self.c.retrain_steps == 0:
                    self.optimizer.update_model(self.cumulative_steps, self.c.batch_size,
                                                mini_iter=self.c.train_mini_epochs)

                state = next_state
                action = next_action
                episode_reward += reward[0, 0]
                self.cumulative_steps += 1

                if render:
                    self.env.render()
                if done:
                    break

            print(f'episode: {n_rollouts:>4} \t reward: {episode_reward:>14.8f}')
            rewards.append([self.cumulative_steps, episode_reward])
            n_rollouts += 1


class IMPPI1(Agent):
    def __init__(self, config: Union[str, Dict, None], model_path: str):
        super(IMPPI1, self).__init__(config, model_path)

        self.first_run = True
        self.cumulative_steps = 0

    def _load_config(self, config: Dict):
        self.c = load_dataclass_from_dict(IMPPIConfig, config)

    def _init_env(self):
        super(IMPPI1, self)._init_env()
        # self.env = gym.wrappers.Monitor(self.env, '/tmp/mppi/', force=True)
        self.env.reset()
        if self.c.env_init_state:
            # self.env.env.state = self.c.env_init_state
            self.env.state = self.c.env_init_state

    def _init_agent(self, config: Dict):
        """ initialize buffer, dynamics model and MPPI policy """
        self.buffer = ReplayBuffer(self.c.buffer_size, a_dim=self.a_dim, s_dim=self.s_dim, device=self.device)
        self._setup_ac(config)
        self._setup_twin_env(config)
        self._init_policy(config)

    def _setup_twin_env(self, config):
        # setup dynamics model
        twin_env_config = config['twin_env']
        twin_env_config["s_dim"] = self.s_dim
        twin_env_config["a_dim"] = self.a_dim
        assert self.c.twin_env_type != ""
        self.twin_env = TwinEnvFactory.create_model(self.c.twin_env_type, twin_env_config, self.model_path, self.device)

        dynamics_train_config = config['dynamics_train']
        dynamics_train_config['max_steps'] = self.c.max_steps
        dynamics_train_config['bootstrap_steps'] = self.c.bootstrap_steps
        self.optimizer = DynamicsModelOptimizer(
            dynamics=self.twin_env,
            replay_buffer=self.buffer,
            **dynamics_train_config
        )

    def _setup_ac(self, config):
        # setup ac model
        ac_model_config = config['ac_model']
        ac_model_config["s_dim"] = self.s_dim
        ac_model_config["a_dim"] = self.a_dim
        self.ac = TwinEnvFactory.create_model(self.c.ac_type, ac_model_config, self.model_path, self.device)
        self.optimizer2 = CriticOptimizer(
            critic=self.ac.critic,
            replay_buffer=self.buffer,
        )

    def _init_policy(self, config: Dict, **kwargs):
        super(IMPPI1, self)._init_policy(
            config,
            twin_env=self.twin_env,
            ac=self.ac,
        )

    def _get_action(self, state):
        return self.policy.forward(state)

    def policy_learn(self, state):
        return self.policy.update_action(state)

    def learn(self):
        try:
            self.train()
        except KeyboardInterrupt:
            logging.warning("Interrupt training")
            # self.env.stats_recorder.save_complete()
            # self.env.stats_recorder.done = True

    def train(self):
        rewards = []

        n_rollouts = 1
        while self.cumulative_steps < self.c.max_steps:
            state = self.env.reset()
            action = self._get_action(state)
            render = n_rollouts % self.c.render_every_n_rollout == 0
            episode_reward = 0
            for step in range(1, self.c.rollout_steps + 1):
                next_state, reward, done, infos = self.env.step(action.copy().squeeze())
                #self.policy_learn(state)
                next_action = self._get_action(next_state)
                self.buffer.add(state, action, reward, next_state.squeeze(), done, infos)

                if self.cumulative_steps == self.c.bootstrap_steps or \
                        self.cumulative_steps > self.c.bootstrap_steps and \
                        self.cumulative_steps % self.c.retrain_steps == 0:
                    self.optimizer.update_model(self.cumulative_steps, self.c.batch_size,
                                                mini_iter=self.c.train_mini_epochs)
                    self.optimizer2.update_model(self.c.batch_size, mini_iter=self.c.train_mini_epochs)
                    self.policy_learn(state)

                state = next_state
                action = next_action
                episode_reward += reward[0, 0]
                self.cumulative_steps += 1

                if render:
                    self.env.render()
                if done:
                    break

            print(f'episode: {n_rollouts:>4} \t reward: {episode_reward:>14.8f}')
            rewards.append([self.cumulative_steps, episode_reward])
            n_rollouts += 1