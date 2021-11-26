import os
import math
import logging
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Type, Callable, Union, List, Any
from marshmallow_dataclass import dataclass
from dataclasses import field
from marshmallow import validate
# import torch.nn.functional as F
# from torch.distributions import Normal
#from equation_learner import model_factory as dynamic_model_factory
#from neql.equation_learner.eql import ModelFactory
#from learning_dynamics.learn_dynamical_system import model_factory as value_model_factory
from torch.autograd import Variable
from robot_utils.py.utils import load_dataclass
from robot_utils.torch.torch_utils import create_sequential_adaptive_linear
import numpy as np
np.set_printoptions(suppress=False)
# from learning_dynamics.learn_dynamical_system.lyapunov_function import QCNN, ICNN, SPD, QCNNConfig, \
#     ICNNConfig, SPDConfig, TrainConfig, train_model_lya, test_lya_model


@dataclass
class TwinEnvConfig:
    s_dim:          Union[int, List[int]]
    a_dim:          Union[int, List[int]]
    type:           str = field(metadata={"validate": validate.OneOf(["dyn_rew", "dyn+rew", "dyn", "rew", "ac"])})
    network_type:   Union[str, Dict[str, str]]
    network_act:    Union[str, Dict[str, str]] = "ReLU"
    ac_act:         Union[str, Dict[str, str]] = "ReLU"
    dyn_struct:     Union[List[int], None] = None
    ac_struct:      Union[List[int], None] = None
    rew_struct:     Union[List[int], None] = None
    all_struct:     Union[List[int], None] = None


class TwinEnv(nn.Module, ABC):
    def __init__(self, config, path, device):
        super(TwinEnv, self).__init__()
        self._load_config(config)
        self.model_path = os.path.join(path, f"twin_env_{self.name}_{self.c.type}.pt")
        self.device = device
        if self.c.type == "dyn_rew" or self.c.type == "dyn":
            self._init_dynamics()
        if self.c.type == "dyn_rew" or self.c.type == "rew":
            self._init_reward()
        if self.c.type == "dyn+rew":
            self._init_all()
        if self.c.type == "ac":
            self._init_actor()
            self._init_critic()


    @property
    def name(self):
        return self.__class__.__name__

    def _load_config(self, config):
        self.c = load_dataclass(TwinEnvConfig, config)

    def _init_dynamics(self):
        self.dyn = None

    def _init_reward(self):
        self.rew = None

    def _init_actor(self):
        self.actor = None

    def _init_critic(self):
        self.critic = None

    def _init_all(self):
        self.model = None

    def reset_rew(self, rew_func):
        self.rew = rew_func

    @abstractmethod
    def forward(self, s: torch.Tensor, a: torch.Tensor):
        pass

    @abstractmethod
    def step(self, s: torch.Tensor, a: torch.Tensor, **kwargs):
        pass

    @abstractmethod
    def get_logp(self):
        pass


class TwinEnvFactory:
    """ The TwinEnv factory class """

    registry = {}  # type: Dict[str, Type[TwinEnv]]
    """ Internal registry for available models """

    @classmethod
    def register(cls, name: str) -> Callable:
        """ Class method to register sub-class to the internal registry.
        Args:
            name (str): The name of the sub-class.
        Returns:
            The sub-class itself.
        """

        def inner_wrapper(wrapped_class: Type[TwinEnv]) -> Type[TwinEnv]:
            if name in cls.registry:
                logging.warning(f'Model {name} already exists. Will replace it.')
            cls.registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @classmethod
    def create_model(cls,
                     name: str,
                     config_file: Union[str, Dict, None] = None,
                     model_path: str = None,
                     device: torch.device = 'cuda') -> TwinEnv:
        """ Factory method to create the model with the given name and pass parameters as``kwargs``.
        Args:
            name (str): The name of the model to create.
            config_file:
            model_path:
            device:
        Returns:
            An instance of the model that is created.
        """
        # logging.info(f"available model factory {cls.registry}")
        logging.info(f"create model with name {name}")
        if name not in cls.registry:
            raise NotImplementedError(f'Model {name} does not exist in the registry')

        model_class = cls.registry[name]
        model = model_class(config_file, model_path, device)
        return model


@dataclass
class MLPDynConfig(TwinEnvConfig):
    std: float = 0.0


@TwinEnvFactory.register("MLPDyn")
class MLPDyn(TwinEnv):
    """ Dynamics are represented by neural networks, while reward is a callable set from outside """
    def __init__(self, config, path, device):
        super(MLPDyn, self).__init__(config, path, device)

    def _load_config(self, config):
        self.c = load_dataclass(MLPDynConfig, config)
        if isinstance(self.c.network_act, dict):
            self.dyn_act = self.c.network_act.get("dyn", "Tanh")
        else:
            self.dyn_act = self.c.network_act

    def _init_dynamics(self):
        struct = [self.c.a_dim + self.c.s_dim + 1, ] + self.c.dyn_struct + [self.c.s_dim, ]
        self.dyn = create_sequential_adaptive_linear(struct, activation=self.dyn_act).to(self.device)
        self.log_std = nn.Parameter(torch.ones(1, self.c.s_dim) * self.c.std).to(self.device)

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        state = self.step_state(s, a)
        rew = self.rew(s.detach().cpu().numpy(), a.detach().cpu().numpy())
        std = torch.clamp(self.log_std, -20., 2).exp().expand_as(state)
        return state, std, torch.from_numpy(rew).to(state)

    def step_state(self, s: torch.Tensor, a: torch.Tensor, **kwargs):
        assert len(s.shape) == len(a.shape)
        x = torch.cat([s, a], dim=-1)
        x = torch.cat((torch.sin(x[:, 0]).view(-1, 1), torch.cos(x[:, 0]).view(-1, 1), x[:, 1:]), dim=1)
        state = s + self.dyn(x)
        state[:, 0] = self.angle_normalize(state[:, 0])
        return state

    @staticmethod
    def angle_normalize(x):
        return ((x + math.pi) % (2 * math.pi)) - math.pi

    def step(self, s: torch.Tensor, a: torch.Tensor, **kwargs):
        mean, std, rew = self.forward(s, a)
        return mean, rew, False, {}

    def get_logp(self):
        pass


@dataclass
class MLPDynRewConfig(TwinEnvConfig):
    std: float = 0.0


@TwinEnvFactory.register("MLPDynRew")
class MLPDynRew(TwinEnv):
    def __init__(self, config, path, device):
        super(MLPDynRew, self).__init__(config, path, device)

    def _load_config(self, config):
        self.c = load_dataclass(MLPDynRewConfig, config)
        if isinstance(self.c.network_act, dict):
            self.dyn_act = self.c.network_act.get("dyn", "Tanh")
            self.rew_act = self.c.network_act.get("rew", "ReLU")
        else:
            self.rew_act = self.dyn_act = self.c.network_act

    def _init_dynamics(self):
        struct = [self.c.a_dim + self.c.s_dim + 1, ] + self.c.dyn_struct + [self.c.s_dim, ]
        self.dyn = create_sequential_adaptive_linear(struct, activation=self.dyn_act).to(self.device)
        self.log_std = nn.Parameter(torch.ones(1, self.c.s_dim) * self.c.std).to(self.device)

    def _init_reward(self):
        struct = [self.c.a_dim + self.c.s_dim, ] + self.c.rew_struct + [1, ]
        self.rew = create_sequential_adaptive_linear(struct, activation=self.rew_act).to(self.device)

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        assert len(s.shape) == len(a.shape)
        x = torch.cat([s, a], dim=-1)
        rew = self.rew(x)
        xu = torch.cat((torch.sin(x[:, 0]).view(-1, 1), torch.cos(x[:, 0]).view(-1, 1), x[:, 1:]), dim=1)
        s = s + self.dyn(xu)
        s[:, 0] = self.angle_normalize(s[:, 0])
        std = torch.clamp(self.log_std, -20., 2).exp().expand_as(s)
        return s, std, rew

    @staticmethod
    def angle_normalize(x):
        return ((x + math.pi) % (2 * math.pi)) - math.pi

    def step(self, s: torch.Tensor, a: torch.Tensor, **kwargs):
        mean, std, rew = self.forward(s, a)
        return mean, rew, False, {}

    def get_logp(self):
        pass

@dataclass
class MLPACConfig(TwinEnvConfig):
    std: float = 0.0

@TwinEnvFactory.register("MLPAC")
class MLPAC(TwinEnv):
    def __init__(self, config, path, device):
        super(MLPAC, self).__init__(config, path, device)

    def _load_config(self, config):
        self.c = load_dataclass(MLPACConfig, config)
        if isinstance(self.c.ac_act, dict):
            self.act_act = self.c.ac_act.get("act", "ReLU")
            self.cri_act = self.c.ac_act.get("cri", "ReLU")
        else:
            self.act_act = self.cri_act = self.c.ac_act

    def _init_actor(self):
        struct = [self.c.s_dim, ] + self.c.ac_struct + [self.c.a_dim, ]
        self.actor = create_sequential_adaptive_linear(struct, activation=self.act_act).to(self.device)

    def _init_critic(self):
        struct = [self.c.s_dim, ] + self.c.ac_struct + [1, ]
        self.critic = create_sequential_adaptive_linear(struct, activation=self.cri_act).to(self.device)

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        pass

    def step(self, s: torch.Tensor, a: torch.Tensor, **kwargs):
        pass

    def get_logp(self):
        pass

    def go(self, s: torch.Tensor):
        action = self.actor(s)
        value = self.critic(s)
        return action, value

# TODO, try equation learner as dynamics



# class EQLModel(nn.Module):
#     def __init__(self, s_dim, a_dim, config, device, env_name=None):
#
#         super(EQLModel, self).__init__()
#         self.s_dim = s_dim
#         self.a_dim = a_dim
#         self.device = device
#
#         model_config, reward_config = config['model'],          config['reward']
#         model_type,   reward_type   = config['model']['type'],  config['reward']['type']
#         path = "./trained_model/env_{}-model_{}-reward_{}".format(env_name, model_type, reward_type)
#
#         if model_type == "eql":
#             ic("eql model")
#             model_config['in_feats'] = s_dim + a_dim
#             model_config['structures'][-1]['ide'] = s_dim
#         elif model_type == 'fnn':
#             model_config['structure'] = [s_dim + a_dim] + model_config['structure'] + [s_dim]
#
#         ic(model_config)
#         dynamic_model = dynamic_model_factory(model_type, train=False)
#         self.mu = dynamic_model(model_config, path).to(device)
#
#         reward_model = value_model_factory(reward_type, train=False)
#         reward_config['structure'] = [s_dim + a_dim] + reward_config['structure'] + [1]
#         self.reward_fun = reward_model(reward_config, path).to(device)
#
#         std = model_config['std']
#         self.log_std = nn.Parameter(torch.ones((1, s_dim), device=device) * std, requires_grad=True)
#
#     def forward(self, s: torch.Tensor, a: torch.Tensor):
#         state = torch.cat([s, a], dim=1)
#         x, _ = self.mu(state)
#         if isinstance(x, tuple):
#             x, pen_div = x[0], x[1]
#         else:
#             pen_div = Variable(torch.tensor([0.], device=x.device).float())
#
#         std = torch.clamp(self.log_std, -20., 2).exp().expand_as(x)
#         # return x, std, self.reward_fun(torch.cat([s, a], axis=1)), pen_div
#         rew, _ = self.reward_fun(state)
#         return s+x, std, rew, pen_div
#
#     def step(self, x, u):
#         # Note: first try: deterministic model, later try probablistic model
#         mean, std, rew, pen_div = self.forward(x, u)
#         return mean, rew - pen_div
#
#
# class MDNModel(nn.Module):
#     def __init__(self, num_states, num_actions, def_layers=None, std=0.):
#         super(MDNModel, self).__init__()
#         pass
#
#     def forward(self, x, s):
#         pass
#
#
# class DynamicsModel_(nn.Module):
#     """
#     dynamic model with value functions
#     """
#     def __init__(self, s_dim, a_dim, config, device, env_name=None):
#
#         super(DynamicsModel_, self).__init__()
#         self.s_dim = s_dim
#         self.a_dim = a_dim
#         self.device = device
#
#         model_config, reward_config = config['model'],          config['reward']
#         model_type,   reward_type   = config['model']['type'],  config['reward']['type']
#         path = "./trained_model/env_{}-model_{}-reward_{}".format(env_name, model_type, reward_type)
#
#         if model_type == "eql":
#             ic("eql model")
#             model_config['in_feats'] = self.s_dim + self.a_dim
#             model_config['structures'][-1]['ide'] = self.s_dim
#         elif model_type == 'fnn':
#             model_config['structure'] = [self.s_dim + self.a_dim] + model_config['structure'] + [self.s_dim]
#
#         dynamic_model = dynamic_model_factory(model_type, train=False)
#         self.mu = dynamic_model(model_config, path).to(device)
#
#         value_model = value_model_factory(reward_type, train=False)
#         reward_config['structure'] = [self.s_dim + self.a_dim] + reward_config['structure'] + [1]
#         self.value_func = value_model(reward_config, path).to(device)
#
#         std = model_config['std']
#         self.log_std = nn.Parameter(torch.ones((1, self.s_dim), device=device) * std, requires_grad=True)
#
#     def forward(self, s: torch.Tensor, a: torch.Tensor):
#         next_state, std, pen_div = self.forward_dynamics(s, a)
#         value = self.forward_value(torch.cat([s, a], dim=1))
#         return next_state, std, value, pen_div
#
#     def forward_dynamics(self, s: torch.Tensor, a: torch.Tensor):
#         state = torch.cat([s, a], dim=1)
#         x, _ = self.mu(state)
#         if isinstance(x, tuple):
#             x, pen_div = x[0], x[1]
#         else:
#             pen_div = Variable(torch.tensor([0.], device=x.device).float())
#
#         std = torch.clamp(self.log_std, -20., 2).exp().expand_as(x)
#         return s + x, std, pen_div
#
#     def forward_value(self, state: torch.Tensor):
#         value, _ = self.value_func(state)
#         return value
#
#     def step(self, x, u):
#         # Note: first try: deterministic model, later try probablistic model
#         next_state, std, value, pen_div = self.forward(x, u)
#         return next_state, value
#
#
# class DynamicsModel(nn.Module):
#     """
#     dynamic model, very straightforward
#     """
#     def __init__(self, s_dim, a_dim, config, device, env_name=None):
#         super(DynamicsModel, self).__init__()
#
#         self.s_dim = s_dim
#         self.a_dim = a_dim
#         self.device = device
#
#         self.ACTION_LOW = -2.0
#         self.ACTION_HIGH = 2.0
#
#         model_type = config['type']
#         path = "./trained_model/env_{}-model_{}".format(env_name, model_type)
#
#         if model_type == "eql":
#             ic("eql model")
#             config['in_feats'] = self.s_dim + self.a_dim
#             config['structures'][-1]['ide'] = self.s_dim
#         elif model_type == 'fnn':
#             config['structure'] = [self.s_dim + self.a_dim] + config['structure'] + [self.s_dim]
#
#         dynamic_model = dynamic_model_factory(model_type, train=False)
#         self.mu = dynamic_model(config, path).to(device)
#
#         # std = model_config['std']
#         # self.log_std = nn.Parameter(torch.ones((1, self.s_dim), device=device) * std, requires_grad=True)
#
#     def forward(self, s: torch.Tensor, a: torch.Tensor):
#         a = torch.clamp(a, self.ACTION_LOW, self.ACTION_HIGH)
#         state = torch.cat([s, a], dim=1)
#         x, _ = self.mu(state)
#         if isinstance(x, tuple):
#             x, pen_div = x[0], x[1]
#         else:
#             pen_div = Variable(torch.tensor([0.], device=x.device).float())
#
#         # std = torch.clamp(self.log_std, -20., 2).exp().expand_as(x)
#         # return s + x, std, pen_div
#         return s + x
#
#     def step(self, x, u):
#         # Note: first try: deterministic model, later try probablistic model
#         next_state, std, pen_div = self.forward(x, u)
#         return next_state
#

class DynamicsModel1(nn.Module):
    """
    dynamic model, entend the state to cos(angle), sin(angle)
    """
    def __init__(self, s_dim, a_dim, config, device, path, env_name=None):
        super(DynamicsModel1, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.device = device

        self.ACTION_LOW = -2.0
        self.ACTION_HIGH = 2.0

        model_type = config['network_type']

        if model_type == "EQL":
            logging.info("eql model")
            config['in_feats'] = self.s_dim + self.a_dim
            config['structures'][-1]['ide'] = self.s_dim
        elif model_type == 'FNN':
            config['structure'] = [self.s_dim + self.a_dim + 1] + config['structure'] + [self.s_dim]

        self.mu = ModelFactory.create_model(model_type, config, path).to(device)

    @staticmethod
    def angle_normalize(x):
        return ((x + math.pi) % (2 * math.pi)) - math.pi

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        xu = torch.cat((s, a), dim=1)
        # feed in cosine and sine of angle instead of theta
        xu = torch.cat((torch.sin(xu[:, 0]).view(-1, 1), torch.cos(xu[:, 0]).view(-1, 1), xu[:, 1:]), dim=1)
        state_residual, _ = self.mu(xu)
        if isinstance(state_residual, tuple):
            state_residual, pen_div = state_residual[0], state_residual[1]
        # output is delta_theta, so we can directly add it to previous state s
        next_state = s + state_residual
        next_state[:, 0] = self.angle_normalize(next_state[:, 0])
        return next_state
