import torch
import time
import gym
import functools
import numpy as np

from marshmallow_dataclass import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable

from torch.distributions.multivariate_normal import MultivariateNormal
from robot_policy.mpc.mpc_common.twin_env import DynamicsModel, DynamicsModel1, TwinEnv
from robot_policy.rl.common.base_policy import PolicyFactory, BasePolicy, PolicyConfig
from robot_utils.py.utils import load_dataclass_from_dict


def is_tensor_like(x):
    return torch.is_tensor(x) or type(x) is np.ndarray


# from arm_pytorch_utilities, standalone since that package is not on pypi yet
def handle_batch_input(func):
    """For func that expect 2D input, handle input that have more than 2 dimensions by flattening them temporarily"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # assume inputs that are tensor-like have compatible shapes and is represented by the first argument
        batch_dims = []
        for arg in args:
            if is_tensor_like(arg) and len(arg.shape) > 2:
                batch_dims = arg.shape[:-1]  # last dimension is type dependent; all previous ones are batches
                break
        # no batches; just return normally
        if not batch_dims:
            return func(*args, **kwargs)

        # reduce all batch dimensions down to the first one
        args = [v.view(-1, v.shape[-1]) if (is_tensor_like(v) and len(v.shape) > 2) else v for v in args]
        ret = func(*args, **kwargs)
        # restore original batch dimensions; keep variable dimension (nx)
        if type(ret) is tuple:
            ret = [v if (not is_tensor_like(v) or len(v.shape) == 0) else (
                v.view(*batch_dims, v.shape[-1]) if len(v.shape) == 2 else v.view(*batch_dims)) for v in ret]
        else:
            if is_tensor_like(ret):
                if len(ret.shape) == 2:
                    ret = ret.view(*batch_dims, ret.shape[-1])
                else:
                    ret = ret.view(*batch_dims)
        return ret

    return wrapper


class MPPI:
    """
    Model Predictive Path Integral control: iMPPI:  https://github.com/ferreirafabio/mppi_pendulum
    """
    def __init__(self,
                 dynamics: DynamicsModel1,
                 time_dependency=False,
                 running_reward=None,
                 terminal_state_cost=None,
                 actions_per_update=1,
                 horizon=15,
                 num_rollouts=100,
                 samples_per_rollout=1,
                 device="cpu",
                 lambda_=1.,
                 noise_mu=None,
                 noise_sigma=None,
                 a_min=None,
                 a_max=None,
                 a_init=None,
                 a_scale=1,
                 rollout_var_cost=0,
                 rollout_cost_discount=1.0,
                 rollout_var_discount=0.95,
                 sample_null_action=False):

        self.device = device
        self.n_rollouts = num_rollouts
        self.horizon = horizon

        # dimensions of state and control
        self.s_dim = dynamics.s_dim
        self.a_dim = dynamics.a_dim
        self.lambda_ = lambda_

        if noise_mu is None:
            noise_mu = torch.zeros(self.a_dim)
        self.noise_mu = noise_mu.to(self.device)
        self.noise_sigma = torch.tensor(noise_sigma, device=self.device).float()
        # handle 1D edge case
        if self.a_dim == 1:
            self.noise_mu = self.noise_mu.view(-1)
            self.noise_sigma = self.noise_sigma.view(-1, 1)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)

        if a_init is None:
            self.a_init = torch.zeros_like(noise_mu, device=self.device).float()
        else:
            self.a_init = torch.tensor(a_init, device=self.device).float()

        # bounds
        self.a_min = torch.tensor(a_min, device=self.device).float()
        self.a_max = torch.tensor(a_max, device=self.device).float()
        self.a_scale = a_scale
        self.actions_per_update = actions_per_update
        self.action = self.noise_dist.sample((self.horizon,))  # (horizon x a_dim) control sequence

        self.time_dependency = time_dependency
        self.dynamics = dynamics

        self.running_reward = running_reward
        self.terminal_state_cost = terminal_state_cost
        self.sample_null_action = sample_null_action
        self.state = None

        # handling dynamics models that output a distribution (take multiple trajectory samples)
        self.samples_per_rollout = samples_per_rollout
        self.rollout_var_cost = rollout_var_cost
        self.rollout_cost_discount = rollout_cost_discount
        self.rollout_var_discount = rollout_var_discount

    @handle_batch_input
    def _dynamics(self, s: torch.Tensor, a: torch.Tensor, t: float):
        return self.dynamics(s, a, t) if self.time_dependency else self.dynamics(s, a)

    @handle_batch_input
    def _running_cost(self, s: torch.Tensor, a: torch.Tensor):
        return -self.running_reward(s, a)

    def reset(self):
        """
        Clear controller state after finishing a trial
        """
        self.action = self.noise_dist.sample((self.horizon,))

    def update_action(self, state):
        """
        find the next best action
        Args:
            state: array of dimension (s_dim, ) or (num_rollouts, s_dim)

        Returns: array of dimension (a_dim, ), the best action to take

        """
        # shift action sequence 1 time step and initialize the last element
        self.action = torch.roll(self.action, -1, dims=0)                           # (horizon, 1)
        self.action[-1] = self.a_init

        if not torch.is_tensor(state):
            state = torch.tensor(state, device=self.device).float()
        self.state = state

        cost_total = self._compute_total_cost_batch()                               # (num_rollouts, )

        beta = torch.min(cost_total)
        cost_total_non_zero = torch.exp(-1 / self.lambda_ * (cost_total - beta))    # (num_rollouts, )

        eta = torch.sum(cost_total_non_zero)
        omega = (1. / eta) * cost_total_non_zero                                    # (num_rollouts, )
        for t in range(self.horizon):
            self.action[t] += torch.sum(omega.view(-1, 1) * self.noise[:, t], dim=0)  # weighted sum of exploration noise over n_sample rollouts
        action = self.action[:self.actions_per_update]  # self.action (horizon, 1), action (actions_per_update, 1)
        # reduce dimensionality if we only need the first command
        if self.actions_per_update == 1:
            action = action[0]  # (1, )

        return action.detach().cpu().numpy()

    def _compute_total_cost_batch(self) -> torch.Tensor:                                        # (num_rollouts, )
        # parallelize sampling across trajectories
        # resample noise each time we take an action
        self.noise = self.noise_dist.sample((self.n_rollouts, self.horizon))                    # (num_rollouts, horizon, a_dim)
        self.perturbed_action = self.action + self.noise                                        # (num_rollouts, horizon, a_dim)
        if self.sample_null_action:
            self.perturbed_action[self.n_rollouts - 1] = 0
        # naively bound control
        self.perturbed_action = torch.max(torch.min(self.perturbed_action, self.a_max), self.a_min)
        # bounded noise after bounding (some got cut off, so we don't penalize that in action cost)
        self.noise = self.perturbed_action - self.action
        action_cost = self.lambda_ * self.noise @ self.noise_sigma_inv                          # (num_rollouts, horizon, a_dim)

        cost_total, states, actions = self._compute_rollout_costs(self.perturbed_action)        # (num_rollouts, )
        actions /= self.a_scale

        # action perturbation cost (num_rollouts, horizon, 1) x (num_rollouts, horizon, 1)
        perturbation_cost = torch.sum(self.perturbed_action * action_cost, dim=(1, 2))          # (num_rollouts, )
        cost_total += perturbation_cost                                                         # (num_rollouts, )
        return cost_total

    def _compute_rollout_costs(self, perturbed_actions):
        num_rollouts, horizon, a_dim = perturbed_actions.shape
        assert a_dim == self.a_dim

        cost_total = torch.zeros(num_rollouts, device=self.device)                              # (num_rollouts, )
        cost_samples = cost_total.repeat(self.samples_per_rollout, 1)                           # (samples_per_rollout, num_rollouts)
        cost_var = torch.zeros_like(cost_total)                                                 # (num_rollouts, )

        # allow propagation of a sample of states (ex. to carry a distribution), or to start with a single state
        if self.state.shape == (num_rollouts, self.s_dim):
            state = self.state
        else:
            state = self.state.view(1, -1).repeat(num_rollouts, 1)                              # (num_rollouts, s_dim)

        # rollout action trajectory M times to estimate expected cost
        state = state.repeat(self.samples_per_rollout, 1, 1)                                    # (samples_per_rollout, n_sample, s_dim)

        states = []
        actions = []
        for t in range(horizon):
            # note: actions for all samples per rollout are the same, but the dynamics can be probabilistic
            a = self.a_scale * perturbed_actions[:, t].repeat(self.samples_per_rollout, 1, 1)   # (samples_per_rollout, num_rollouts, a_dim)
            state = self._dynamics(state, a, t)                                                 # (samples_per_rollout, num_rollouts, s_dim)
            c = self._running_cost(state, a)                                                    # (samples_per_rollout, num_rollouts, 1)
            cost_samples += (self.rollout_cost_discount ** t) * torch.squeeze(c, dim=-1)        # (samples_per_rollout, num_rollouts)
            if self.samples_per_rollout > 1:
                cost_var += c.var(dim=0) * (self.rollout_var_discount ** t)

            # Save total states/actions
            states.append(state)
            actions.append(a)

        actions = torch.stack(actions, dim=-2)                                                  # (samples_per_rollout, num_rollouts, horizon, a_dim)
        states = torch.stack(states, dim=-2)                                                    # (samples_per_rollout, num_rollouts, horizon, s_dim)

        # action perturbation cost
        if self.terminal_state_cost:
            c = self.terminal_state_cost(states, actions)
            cost_samples += c
        cost_total += cost_samples.mean(dim=0)                                                  # (num_rollouts, )
        cost_total += cost_var * self.rollout_var_cost                                          # (num_rollouts, )
        return cost_total, states, actions

    def get_rollouts(self, state: torch.Tensor, num_rollouts=1):
        """
        Given the initial states, generate num_rollouts trajectories
        Args:
            state: either s_dim vector or (num_rollouts, s_dim) vector for sampled initial states
            num_rollouts: Number of rollouts with same action sequence - for generating samples with stochastic
                          dynamics

        Returns: (num_rollouts, horizon, s_dim) tensor of state trajectories

        """
        state = state.view(-1, self.s_dim)
        if state.size(0) == 1:
            state = state.repeat(num_rollouts, 1)

        horizon = self.action.shape[0]
        states = torch.zeros((num_rollouts, horizon + 1, self.s_dim), device=self.action.device)
        states[:, 0] = state
        for t in range(horizon):
            states[:, t + 1] = self._dynamics(states[:, t].view(num_rollouts, -1),
                                              self.a_scale * self.action[t].view(num_rollouts, -1), t)
        return states[:, 1:]

    def save_model(self):
        self.dynamics.save_model()


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
    Model Predictive Path Integral control: iMPPI:  https://github.com/ferreirafabio/mppi_pendulum
    """
    def __init__(
            self,
            env: gym.Env,
            config: Optional[Dict],
            model_path: str,
            device: torch.device,
            dynamics: DynamicsModel1,
            running_reward=None,
            terminal_state_cost=None
    ):
        super(IMPPIPolicy, self).__init__(env, config, model_path, device)
        self.dynamics = dynamics

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

        self.running_reward = running_reward
        self.terminal_state_cost = terminal_state_cost
        self.state = None

    def _load_config(self, config: Dict):
        self.c = load_dataclass_from_dict(IMPPIConfig, config)

    def _init_policy_networks(self):
        pass

    @handle_batch_input
    def _dynamics(self, s: torch.Tensor, a: torch.Tensor, t: float):
        return self.dynamics(s, a, t) if self.c.time_dependency else self.dynamics(s, a)

    @handle_batch_input
    def _running_cost(self, s: torch.Tensor, a: torch.Tensor):
        return -self.running_reward(s, a)

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
        self.state = state.float()                                                                      # (S, )

        cost_total = self._compute_total_cost_batch()                                                   # (R, )

        beta = torch.min(cost_total)
        cost_total_non_zero = torch.exp(-1 / self.c.lamb * (cost_total - beta))                         # (R, )

        eta = torch.sum(cost_total_non_zero)
        omega = (1. / eta) * cost_total_non_zero                                                        # (R, )
        for t in range(self.c.horizon):
            self.action[t] += torch.sum(omega.view(-1, 1) * self.noise[:, t], dim=0)  # weighted sum of exploration noise over n_sample rollouts
        action = self.action[:self.c.actions_per_update]  # self.action (T, 1), action (actions_per_update, 1)
        # reduce dimensionality if we only need the first command
        if self.c.actions_per_update == 1:
            action = action[0]  # (1, )

        return action

    def _compute_total_cost_batch(self) -> torch.Tensor:                                                # (R, )
        # parallelize sampling across trajectories
        # resample noise each time we take an action
        self.noise = self.noise_dist.sample((self.c.num_rollouts, self.c.horizon))                      # (R, T, A)
        self.perturbed_action = self.action + self.noise                                                # (R, T, A)
        if self.c.sample_null_action:
            self.perturbed_action[self.c.num_rollouts - 1] = 0
        # naively bound control
        self.perturbed_action = torch.max(torch.min(self.perturbed_action, self.a_max), self.a_min)
        # bounded noise after bounding (some got cut off, so we don't penalize that in action cost)
        self.noise = self.perturbed_action - self.action
        action_cost = self.c.lamb * self.noise @ self.noise_sigma_inv                                   # (R, T, A)

        cost_total, states, actions = self._compute_rollout_costs(self.perturbed_action)                # (R, )
        actions /= self.c.a_scale

        # action perturbation cost (num_rollouts, horizon, 1) x (num_rollouts, horizon, 1)
        perturbation_cost = torch.sum(self.perturbed_action * action_cost, dim=(1, 2))                  # (R, )
        cost_total += perturbation_cost                                                                 # (R, )
        return cost_total

    def _compute_rollout_costs(self, perturbed_actions):
        num_rollouts, horizon, a_dim = perturbed_actions.shape
        assert a_dim == self.c.a_dim

        cost_total = torch.zeros(num_rollouts, device=self.device)                                      # (R, )
        cost_var = torch.zeros_like(cost_total)                                                         # (R, )
        cost_samples = cost_total.repeat(self.c.samples_per_rollout, 1)                                 # (N, R)

        # allow propagation of a sample of states (ex. to carry a distribution), or to start with a single state
        if self.state.shape == (num_rollouts, self.c.s_dim):
            state = self.state
        else:
            state = self.state.view(1, -1).repeat(num_rollouts, 1)                                      # (R, S)

        # rollout action trajectory N times to estimate expected cost
        state = state.repeat(self.c.samples_per_rollout, 1, 1)                                          # (R*N, S)

        states = []
        actions = []
        for t in range(horizon):
            # note: actions for all samples per rollout are the same, but the dynamics can be probabilistic
            a = self.c.a_scale * perturbed_actions[:, t].repeat(self.c.samples_per_rollout, 1, 1)       # (N, R, A)
            state = self._dynamics(state, a, t)                                                         # (N, R, s_dim)
            c = self._running_cost(state, a)                                                            # (N, R, 1)
            cost_samples += (self.c.rollout_cost_discount ** t) * torch.squeeze(c, dim=-1)              # (N, R)
            if self.c.samples_per_rollout > 1:
                cost_var += c.var(dim=0) * (self.c.rollout_var_discount ** t)

            # Save total states/actions
            states.append(state)
            actions.append(a)

        actions = torch.stack(actions, dim=-2)                                                          # (N, R, T, A)
        states = torch.stack(states, dim=-2)                                                            # (N, R, T, S)

        # action perturbation cost
        if self.terminal_state_cost:
            c = self.terminal_state_cost(states, actions)
            cost_samples += c
        cost_total += cost_samples.mean(dim=0)                                                          # (R, )
        cost_total += cost_var * self.c.rollout_var_cost                                                # (R, )
        return cost_total, states, actions

    def get_rollouts(self, state: torch.Tensor, num_rollouts=1):
        """
        Given the initial states, generate num_rollouts trajectories
        Args:
            state: either s_dim vector or (num_rollouts, s_dim) vector for sampled initial states
            num_rollouts: Number of rollouts with same action sequence - for generating samples with stochastic
                          dynamics

        Returns: (num_rollouts, horizon, s_dim) tensor of state trajectories

        """
        state = state.view(-1, self.c.s_dim)
        if state.size(0) == 1:
            state = state.repeat(num_rollouts, 1)

        horizon = self.action.shape[0]
        states = torch.zeros((num_rollouts, horizon + 1, self.c.s_dim), device=self.action.device)
        states[:, 0] = state
        for t in range(horizon):
            states[:, t + 1] = self._dynamics(states[:, t].view(num_rollouts, -1),
                                              self.c.a_scale * self.action[t].view(num_rollouts, -1), t)
        return states[:, 1:]

    def save_model(self, best_param=None):
        torch.save(self.state_dict(), self.policy_params_file)

    def load_model(self):
        self.load_state_dict(torch.load(self.policy_params_file))