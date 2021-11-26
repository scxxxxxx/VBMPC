import torch
import torch.nn.functional as F
import gym
import numpy as np
from marshmallow_dataclass import dataclass
from typing import Dict, List, Optional, Callable, Union

from torch.distributions.multivariate_normal import MultivariateNormal
from robot_policy.mpc.mpc_common.twin_env import TwinEnv
from robot_policy.rl.common.base_policy import PolicyFactory, BasePolicy, PolicyConfig
from robot_utils.py.utils import load_dataclass_from_dict


@dataclass
class IMPPIConfig(PolicyConfig):
    # model config
    time_dependency:        bool = False
    actions_per_update:     int = 1
    horizon:                int = 15
    num_rollouts:           int = 100
    samples_per_rollout:    int = 1
    lamb:                   float = 1.
    noise_mu:               Optional[List[float]] = None
    noise_sigma:            Optional[List[float]] = None
    a_init:                 Optional[List[float]] = None
    a_scale:                float = 1

    # handling dynamics models that output a distribution (take multiple trajectory samples)
    rollout_var_cost:       float = 0
    rollout_cost_discount:  float = 1.0
    rollout_var_discount:   float = 0.95
    sample_null_action:     bool = False


@PolicyFactory.register("IMPPIPolicy")
class IMPPIPolicy(BasePolicy):
    """
    Model Predictive Path Integral control
    """
    def __init__(
            self,
            env:           gym.Env,
            config:        Optional[Dict],
            model_path:    str,
            device:        torch.device,
            twin_env:      TwinEnv,
            terminal_cost: Union[Callable, None] = None
    ):
        """
        R:  num_rollouts
        N:  samples_per_rollout
        T:  horizon
        S:  s_dim
        A:  a_dim
        """
        super(IMPPIPolicy, self).__init__(env, config, model_path, device)
        self.twin_env = twin_env

        self.noise_mu = torch.zeros(self.c.a_dim) if self.c.noise_mu is None else torch.tensor(self.c.noise_mu)
        self.noise_mu = self.noise_mu.float().to(self.device)
        self.noise_sigma = torch.tensor(self.c.noise_sigma, device=self.device).float()

        # handle 1D edge case
        if self.c.a_dim == 1:
            self.noise_mu = self.noise_mu.view(-1)
            self.noise_sigma = self.noise_sigma.view(-1, 1)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)

        if self.c.a_init is None:
            self.a_init = torch.zeros_like(self.c.noise_mu, device=self.device).float()
        else:
            self.a_init = torch.tensor(self.c.a_init, device=self.device).float()

        # bounds
        self.a_min = torch.tensor(self.c.a_min, device=self.device).float()
        self.a_max = torch.tensor(self.c.a_max, device=self.device).float()
        self.action = self.noise_dist.sample((self.c.horizon,))                             # (T, A) control sequence

        self.terminal_cost = terminal_cost
        self.state = None

    def _load_config(self, config: Dict):
        self.c = load_dataclass_from_dict(IMPPIConfig, config)

    def _init_policy_networks(self):
        pass

    def reset(self):
        """
        Clear controller state after finishing a trial
        """
        self.action = self.noise_dist.sample((self.c.horizon,))

    def forward(self, obs: torch.Tensor) -> np.ndarray:
        action = self.update_action(obs)
        return action.detach().cpu().numpy()

    def _predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.update_action(obs)

    def update_action(self, state):
        """
        find the next best action
        Args:
            state: array of dimension (s_dim, ) or (num_rollouts, s_dim)

        Returns: array of dimension (a_dim, ), the best action to take
        """
        # shift action sequence 1 time step forward and initialize the last element
        self.action = torch.roll(self.action, -1, dims=0)                                               # (T, A)
        self.action[-1] = self.a_init

        if not torch.is_tensor(state):
            state = torch.tensor(state, device=self.device)
        self.state = state.float()                                                                      # (S, ), (R, S)

        cost_total = self._compute_total_cost_batch()                                                   # (R, )

        beta = torch.min(cost_total)
        cost_total_non_zero = torch.exp(-1 / self.c.lamb * (cost_total - beta))                         # (R, )

        eta = torch.sum(cost_total_non_zero)
        omega = (1. / eta) * cost_total_non_zero                                                        # (R, )
        # weighted sum of exploration noise over R rollouts
        for t in range(self.c.horizon):
            self.action[t] += torch.sum(omega.view(-1, 1) * self.noise[:, t], dim=0)
        action = self.action[:self.c.actions_per_update].squeeze(0)   # (A,) or (self.c.actions_per_update, A)
        return action

    def _compute_total_cost_batch(self) -> torch.Tensor:                                                # (R, )
        self.noise = self.noise_dist.sample((self.c.num_rollouts, self.c.horizon))                      # (R, T, A)
        self.perturbed_action = self.action + self.noise                                                # (R, T, A)
        if self.c.sample_null_action:
            self.perturbed_action[self.c.num_rollouts - 1] = 0
        self.perturbed_action = torch.max(torch.min(self.perturbed_action, self.a_max), self.a_min)
        # bounded noise after bounding (some got cut off, so we don't penalize that in action cost)
        self.noise = self.perturbed_action - self.action
        action_cost = self.c.lamb * self.noise @ self.noise_sigma_inv                                   # (R, T, A)

        cost_total, states, actions = self._compute_rollout_costs(self.perturbed_action)                # (R, )
        actions /= self.c.a_scale

        # action perturbation cost (R, T, A) x (R, T, A)
        perturbation_cost = torch.sum(self.perturbed_action * action_cost, dim=(1, 2))                  # (R, )
        cost_total += perturbation_cost                                                                 # (R, )
        return cost_total

    def _compute_rollout_costs(self, perturbed_actions):                                                # (R, T, A)
        num_rollouts, horizon, a_dim = perturbed_actions.shape
        n = self.c.samples_per_rollout
        batch = n * num_rollouts  # B
        s_dim = self.c.s_dim
        discount = self.c.rollout_cost_discount
        assert a_dim == self.c.a_dim

        cost_total = torch.zeros(num_rollouts, device=self.device)                                      # (R, )
        cost_var = torch.zeros_like(cost_total)                                                         # (R, )
        costs   = torch.zeros((batch, horizon, 1), device=self.device)                                  # (B, T, 1)
        states  = torch.zeros((batch, horizon, s_dim), device=self.device)                              # (B, T, S)
        actions = torch.zeros((batch, horizon, a_dim), device=self.device)                              # (B, T, A)

        # first N rows are the N times repeat of the state for the first rollout, and so on
        if self.state.shape == (num_rollouts, self.c.s_dim):
            state = self.state.repeat_interleave(n, dim=0)                                              # (B, S)
        else:
            state = self.state.view(1, -1).repeat(batch, 1)                                             # (B, S)

        for t in range(horizon):
            # note: actions for all samples per rollout are the same, but the dynamics can be probabilistic
            a = self.c.a_scale * perturbed_actions[:, t].repeat_interleave(n, dim=0)                    # (B,A)
            state, rew, done, info = self.twin_env.step(state, a)                       # (B, S), (B, 1), (B, 1), _
            rew = rew.detach()
            costs[:, t] = -(discount ** t) * rew                                                        # (B, 1)
            if n > 1:
                cost_var += rew.view(num_rollouts, n).var(dim=1) * (self.c.rollout_var_discount ** t)
            states[:, t] = state
            actions[:, t] = a

        if self.terminal_cost:
            c = self.terminal_cost(states[:, -1], actions[:, -1])
            costs[:, -1] += c
        cost_total += costs.squeeze().sum(-1).reshape(num_rollouts, n).mean(dim=1)                      # (R, )
        cost_total += cost_var * self.c.rollout_var_cost                                                # (R, )

        actions = actions.reshape(num_rollouts, n, horizon, -1)                                         # (R, N, T, A)
        states = states.reshape(num_rollouts, n, horizon, -1)                                           # (R, N, T, S)
        return cost_total, states, actions

    # def get_rollouts(self, state: torch.Tensor, num_rollouts=1):
    #     """
    #     Given the initial states, generate num_rollouts trajectories
    #     Args:
    #         state: either s_dim vector or (num_rollouts, s_dim) vector for sampled initial states
    #         num_rollouts: Number of rollouts with same action sequence - for generating samples with stochastic
    #                       dynamics
    #
    #     Returns: (num_rollouts, horizon, s_dim) tensor of state trajectories
    #
    #     """
    #     state = state.view(-1, self.c.s_dim)
    #     if state.size(0) == 1:
    #         state = state.repeat(num_rollouts, 1)
    #
    #     horizon = self.action.shape[0]
    #     states = torch.zeros((num_rollouts, horizon + 1, self.c.s_dim), device=self.action.device)
    #     states[:, 0] = state
    #     for t in range(horizon):
    #         states[:, t + 1] = self._dynamics(states[:, t].view(num_rollouts, -1),
    #                                           self.c.a_scale * self.action[t].view(num_rollouts, -1), t)
    #     return states[:, 1:]

    def save_model(self, best_param=None):
        torch.save(self.state_dict(), self.policy_params_file)

    def load_model(self):
        self.load_state_dict(torch.load(self.policy_params_file))

@dataclass
class VBMPCConfig(PolicyConfig):
    # model config
    time_dependency:        bool = False
    actions_per_update:     int = 1
    horizon:                int = 15
    num_rollouts:           int = 100
    samples_per_rollout:    int = 1
    lamb:                   float = 1.
    a_scale:                float = 1
    lr:                     float = 0.001

    # handling dynamics models that output a distribution (take multiple trajectory samples)
    rollout_var_cost:       float = 0
    rollout_cost_discount:  float = 1.0
    discount:               float = 0.95
    sample_null_action:     bool = False

@PolicyFactory.register("VBMPCPolicy")
class VBMPCPolicy(BasePolicy):
    def __init__(
            self,
            env:           gym.Env,
            config:        Optional[Dict],
            model_path:    str,
            device:        torch.device,
            twin_env:      TwinEnv,
            ac:            TwinEnv,
    ):
        super(VBMPCPolicy, self).__init__(env, config, model_path, device)
        self.twin_env = twin_env
        self.ac = ac
        self.actor_optimizer = torch.optim.Adam(self.ac.actor.parameters(), lr=self.c.lr)
        self.state = None

    def _load_config(self, config: Dict):
        self.c = load_dataclass_from_dict(VBMPCConfig, config)

    def _init_policy_networks(self):
        pass

    def forward(self, obs) -> np.ndarray:
        a, _, _ = self.rollout(obs)
        action = a[0]
        return action.detach().cpu().numpy()

    def _predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        pass

    # def choose_action(self, obs):
    #     obs = torch.tensor(obs, device=self.device)
    #     action, _ = self.ac.go(obs)
    #     # with torch.no_grad():
    #     #     action = F.softmax(network_output, dim=0)
    #     # action = np.random.choice(range(prob_weights.shape[0]), p=prob_weights)
    #     return action

    def rollout(self, obs):
        state = torch.zeros(self.c.horizon+1, self.c.s_dim, device=self.device)
        action = torch.zeros(self.c.horizon, self.c.a_dim, device=self.device)
        value = torch.zeros(self.c.horizon, 1, device=self.device)
        # obs = torch.tensor(obs, device=self.device)
        # state[0, :] = obs.squeeze()
        state[0, :] = torch.tensor(obs, device=self.device).view(1, -1)

        for i in range(self.c.horizon):
            action[i, :], value[i] = self.ac.go(state[i, :])
            state[i+1, :] = self.twin_env.step_state(state[i, :].unsqueeze(dim=0), action[i, :].unsqueeze(dim=0))

        return action, state, value

    def update_action(self, obs):
        actions, states, values = self.rollout(obs)
        # action, value = self.ac.go(state)
        # action = action.detach()
        state = states.detach()
        # value = value.detach()
        action, value = self.ac.go(state)
        terminal_value = (self.c.discount ** self.c.horizon) * value[self.c.horizon-1]

        cost1 = torch.matmul(torch.transpose(state, 0, 1), state)
        running_cost1 = torch.zeros(cost1.shape[0], 1)
        for i in range(cost1.shape[0]):
            running_cost1[i] = (self.c.discount ** i) * cost1[i, i]

        cost2 = torch.matmul(torch.transpose(action, 0, 1), action)
        running_cost2 = torch.zeros(cost2.shape[0], 1)
        for i in range(cost2.shape[0]):
            running_cost2[i] = (self.c.discount ** i) * cost2[i, i]

        running_cost = torch.sum(running_cost1) + torch.sum(running_cost2)

        v = torch.zeros(self.c.horizon, 1)
        for i in range(self.c.horizon):
            v[i] = torch.norm(action[i, :], p=2)/2
        norm = torch.sum(v)

        loss = terminal_value 

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def save_model(self, best_param=None):
        torch.save(self.state_dict(), self.policy_params_file)

    def load_model(self):
        self.load_state_dict(torch.load(self.policy_params_file))

