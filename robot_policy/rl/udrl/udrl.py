import os
import numpy as np
import matplotlib.pyplot as plt
import gym

from collections import namedtuple
from typing import List, Union, Dict, Callable
from marshmallow_dataclass import dataclass
from robot_utils.py.utils import create_path, load_dataclass
from robot_utils.torch.torch_utils import get_device

import torch
import torch.nn.functional as F

from robot_policy.rl.common import EpisodicReplayBuffer
from robot_policy.rl.udrl import Behavior
from robot_env import get_env_from_name
from robot_utils.gym.gym_utils import get_space_dim
from robot_utils.torch.torch_utils import init_torch, set_random_seeds
init_torch()
Tensor = torch.Tensor


# Helper function to create episodes as namedtuple
make_episode = namedtuple('Episode', field_names=[
    'states', 'actions', 'rewards', 'init_command', 'total_return', 'length',
])


@dataclass
class UDRLConfig:
    # environment: RocketLander-v0 | LunarLander-v2 | MountainCar-v0 | CartPole-v0
    env_name:               str = "RocketLander-v0"
    render:                 bool = True
    seed:                   int = 123
    env_init_state:         list = None
    use_gpu:                bool = True

    # policy and training
    horizon_scale:          float = 0.01        # Scaling factor for desired horizon input
    return_scale:           float = 0.02        # Scaling factor for desired return input
    elite_population:       int = 75            # Number of episodes from the end of the replay buffer used for sampling exploratory commands
    learning_rate:          float = 0.0003      # Learning rate for the ADAM optimizer
    hidden_size:            int = 32            # Hidden units

    stop_on_solved:         bool = False        # Will stop the training when the agent gets `target_return` `n_evals` times
    target_return:          int = 200           # Target return before breaking out of the training loop

    # buffer and sampling
    buffer_capacity:        int = 500           # Maximum size of the replay buffer (in episodes)
    exploration_scale:      float = 1.0         # the scaling of tge exploration range of the desired return (sampled from a Gaussian)
    batch_size:             int = 768           # Number of (input, target) pairs per batch used for training the behavior function


    # simulation
    n_main_iter:            int = 700           # Number of iterations in the main loop
    n_episodes_per_iter:    int = 20            # Number of exploratory episodes generated per step of UDRL training
    n_updates_per_iter:     int = 100           # Number of gradient-based updates of the behavior function per step of UDRL training
    n_warm_up_episodes:     int = 10            # Number of warm up episodes at the beginning of training
    n_evals:                int = 1             # Number of episodes that we evaluate the agent
    evaluate_every:         int = 10            # Evaluate the agent after `evaluate_every` iterations

    max_reward:             int = 250           # Maximun reward given by the environment
    max_steps_per_episode:  int = 300           # Maximun steps allowed
    max_steps_reward:       int = -50           # Punish with negative reward when agent reaches 'max_steps`


class UDRLTrainer:
    '''
    Upside-Down Reinforcement Learning main algrithm
    '''
    def __init__(self, config_file: Union[str, Dict, None] = None, model_path: str = None):
        self._load_config(config_file, model_path)
        self._setup_env()
        self._setup_buffer()
        self._setup_actor()

        self.learning_history = []

    def _load_config(self, config_file: Union[str, Dict, None], model_path) -> None:
        self.c = load_dataclass(UDRLConfig, config_file) if config_file else torch.load(self.config_file)
        self.device = get_device(self.c.use_gpu)

        default_path_name = os.path.join('saved_models/', f"{self.c.env_name}")
        self.model_path = model_path if model_path else default_path_name
        create_path(self.model_path)

    def _setup_env(self):
        set_random_seeds(self.c.seed)
        self.env = get_env_from_name(self.c.env_name, self.c.render, make=True)
        # self.env = wrappers.Monitor(self.env, '/tmp/mppi/', force=True)
        self.env.reset()
        if self.c.env_init_state:
            self.env.env.state = self.c.env_init_state

        if isinstance(self.env.action_space, gym.spaces.Box):
            self.a_max = self.env.action_space.high
            self.a_min = self.env.action_space.low

        self.a_dim = get_space_dim(self.env.action_space)
        self.s_dim = get_space_dim(self.env.observation_space)

    def _setup_buffer(self):
        '''
        Initialize the episodic replay buffer with warm-up episodes generated using random actions.
        '''
        self.buffer = EpisodicReplayBuffer(self.c.buffer_capacity)
        random_policy = lambda state, command: np.random.randint(self.a_dim)
        self._generate_episodes(policy=random_policy, n_episodes=self.c.n_warm_up_episodes)

    def _setup_actor(self):
        self.behavior = Behavior(
                self.s_dim,
                self.a_dim,
                self.c.hidden_size,
                [self.c.return_scale, self.c.horizon_scale],
                self.device,
                self.c.learning_rate
            )

    def train(self):
        for i in range(1, self.c.n_main_iter + 1):
            mean_loss = self._train_behavior()
            print('Iter: {}, Loss: {:.4f}'.format(i, mean_loss), end='\r')
            # Sample exploratory commands and generate episodes
            self._generate_episodes(policy=self.behavior.action, n_episodes=self.c.n_episodes_per_iter)

            if i % self.c.evaluate_every == 0:
                command = self._sample_command()
                mean_return = self.evaluate_agent(command)

                self.learning_history.append({
                    'training_loss': mean_loss,
                    'desired_return': command[0],
                    'desired_horizon': command[1],
                    'actual_return': mean_return,
                })

                if self.c.stop_on_solved and mean_return >= self.c.target_return:
                    break

    def _sample_command(self):
        '''Sample a exploratory command: I'd like to gain desired_return in desired_horizon steps'''

        if len(self.buffer) == 0: return [1, 1]

        # 1. sample "last_few" elite episodes
        episodes = self.buffer.get(self.c.elite_population)

        # 2. find the average length over all episodes as desired horizon
        lengths = [episode.length for episode in episodes]
        desired_horizon = round(np.mean(lengths))

        # 3. use Gaussian to approximate the total-return distribution over all elite episodes, and sample the desired return
        returns = [episode.total_return for episode in episodes]
        mean_return, std_return = np.mean(returns), np.std(returns)
        desired_return = np.random.uniform(mean_return, mean_return + std_return * self.c.exploration_scale)

        return [desired_return, desired_horizon]

    def _generate_episodes(self, policy, n_episodes: int):
        '''Generates episodes with exploratory commands sampled from the buffer and actions with defined policy'''

        for i in range(n_episodes):
            command = self._sample_command()
            episode = self._generate_episode(policy, command)
            self.buffer.add(episode)

        self.buffer.sort()

    def _generate_episode(self, policy: Callable[[Tensor, Tensor], Tensor], init_command: List = None):
        '''Generate an episode using the defined policy and start from the initial command
        Returns:
            Namedtuple (states, actions, rewards, init_command, total_return, length)
        '''
        command = [1, 1] if not init_command else init_command.copy()
        desired_return = command[0]
        desired_horizon = command[1]

        states = []
        actions = []
        rewards = []

        time_steps = 0
        done = False
        total_rewards = 0
        state = self.env.reset().tolist()

        while not done:
            # simulate in the environment with current state
            state_input = torch.tensor(state, dtype=torch.float, device=self.device)
            command_input = torch.tensor(command, dtype=torch.float, device=self.device)
            action = policy(state_input, command_input)
            next_state, reward, done, _ = self.env.step(action)

            if not done and time_steps > self.c.max_steps_per_episode:
                done = True
                reward = self.c.max_steps_reward

            # Sparse rewards. Cumulative reward is delayed until the end of each episode
            #         total_rewards += reward
            #         reward = total_rewards if done else 0.0

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state.tolist()

            # Clipped such that it's upper-bounded by the maximum return achievable in the env
            desired_return = min(desired_return - reward, self.c.max_reward)

            # Make sure it's always a valid horizon
            desired_horizon = max(desired_horizon - 1, 1)

            command = [desired_return, desired_horizon]
            time_steps += 1

        return make_episode(states, actions, rewards, init_command, sum(rewards), time_steps)

    def _train_behavior(self):
        '''Training loop

        Params:
            behavior (Behavior)
            buffer (ReplayBuffer)
            n_updates (int):
                how many updates we're gonna perform
            batch_size (int):
                size of the bacth we're gonna use to train on

        Returns:
            float -- mean loss after all the updates
        '''
        all_loss = []
        for update in range(self.c.n_updates_per_iter):
            episodes = self.buffer.random_batch(self.c.batch_size)

            batch_states = []
            batch_commands = []
            batch_actions = []

            for episode in episodes:
                T = episode.length
                t1 = np.random.randint(0, T)
                t2 = np.random.randint(t1 + 1, T + 1)
                desired_reward = sum(episode.rewards[t1:t2])
                desired_horizon = t2 - t1

                state_t1 = episode.states[t1]
                action_t1 = episode.actions[t1]

                batch_states.append(state_t1)
                batch_actions.append(action_t1)
                batch_commands.append([desired_reward, desired_horizon])

            batch_states = torch.tensor(batch_states, dtype=torch.float, device=self.device)
            batch_commands = torch.tensor(batch_commands, dtype=torch.float, device=self.device)
            batch_actions = torch.tensor(batch_actions, dtype=torch.long, device=self.device)

            pred = self.behavior(batch_states, batch_commands)
            loss = F.cross_entropy(pred, batch_actions)

            self.behavior.optim.zero_grad()
            loss.backward()
            self.behavior.optim.step()

            all_loss.append(loss.item())

        return np.mean(all_loss)

    def evaluate_agent(self, command, render=False):
        '''
        Evaluate the agent performance by running an episode
        following Algorithm 2 steps

        Params:
            env (OpenAI Gym Environment)
            behavior (Behavior)
            command (List of float)
            render (bool) -- default False:
                will render the environment to visualize the agent performance
        '''
        self.behavior.eval()
        print('\nEvaluation.', end=' ')

        desired_return = command[0]
        desired_horizon = command[1]
        print('Desired return: {:.2f}, Desired horizon: {:.2f}.'.format(desired_return, desired_horizon), end=' ')

        all_rewards = []
        for e in range(self.c.n_evals):
            done = False
            total_reward = 0
            state = self.env.reset().tolist()

            while not done:
                if render: self.env.render()

                state_input = torch.tensor(state, dtype=torch.float, device=self.device)
                command_input = torch.tensor(command, dtype=torch.float, device=self.device)

                action = self.behavior.greedy_action(state_input, command_input)
                next_state, reward, done, _ = self.env.step(action)

                total_reward += reward
                state = next_state.tolist()

                desired_return = min(desired_return - reward, self.c.max_reward)
                desired_horizon = max(desired_horizon - 1, 1)

                command = [desired_return, desired_horizon]

            if render: self.env.close()
            all_rewards.append(total_reward)

        mean_return = np.mean(all_rewards)
        print('Reward achieved: {:.2f}'.format(mean_return))
        self.behavior.train()
        return mean_return


def plot_history(learning_history):
    desired_return = [h['desired_return'] for h in learning_history]
    desired_horizon = [h['desired_horizon'] for h in learning_history]
    training_loss = [h['training_loss'] for h in learning_history]
    actual_return = [h['actual_return'] for h in learning_history]

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    axs[0, 0].plot(desired_return)
    axs[0, 0].set_title('Desired return')
    axs[0, 1].plot(desired_horizon)
    axs[0, 1].set_title('Desired horizon')
    axs[1, 0].plot(training_loss)
    axs[1, 0].set_title('Training loss')
    axs[1, 1].plot(actual_return)
    axs[1, 1].set_title('Actual return')

    plt.subplots(figsize=(16, 8))
    plt.plot(desired_return)
    plt.plot(actual_return)
    plt.legend(['Desired return', 'Actual return'])


def main():
    udrl = UDRLTrainer(dict(env_name="LunarLander-v2"), model_path=None)
    udrl.train()
    udrl.evaluate_agent([250, 230], render=True)
    plot_history(udrl.learning_history)


if __name__ == "__main__":
    main()
