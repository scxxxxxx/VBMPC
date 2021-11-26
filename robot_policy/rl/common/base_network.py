from typing import Dict, List, Tuple, Type, Union, Optional
from marshmallow_dataclass import dataclass
import numpy as np
import gym
import torch
import torch.nn as nn

# from equation_learner.eql import EQLConfig, EQLTrain, EQL
from robot_utils.gym.gym_utils import get_space_dim
from robot_utils.py.utils import load_dataclass


@dataclass
class NetworkConfig:
    a_dim:      Optional[int] = None
    s_dim:      Optional[int] = None


@dataclass
class MlpNetworkConfig(NetworkConfig):
    activation_fn:      Optional[str] = "ReLU"
    net_struct:         Dict[str, List[int]] = None


class MlpNetwork(nn.Module):
    """
    Maps observation into a latent space, upon which action distribution can be built
    """
    def __init__(self, config: MlpNetworkConfig, device):
        super(MlpNetwork, self).__init__()
        # ic(config)
        self.c = load_dataclass(MlpNetworkConfig, config)
        activation_fn = getattr(torch.nn, self.c.activation_fn)
        # ic(self.c)
        net_struct = self.c.net_struct
        if net_struct is None:
            raise RuntimeError("net_structure is not set"
                               "e.g. {'shared': [3, 64, 64], 'policy': [128, 64], 'value': [32, 32]}")

        if net_struct['shared'] is not None:
            # ic(net_struct['shared'])
            net_struct["shared"].insert(0, self.c.s_dim)
        else:
            net_struct["policy"].insert(0, self.c.s_dim)
            net_struct["value"].insert(0, self.c.s_dim)

        shared_net, policy_net, value_net = [], [], []
        policy_only_layers = []
        value_only_layers = []

        # Iterate through the shared layers and build the shared parts of the network
        # Note: e.g. net_struct = {"shared": [3, 64, 64], "policy": [128, 64], "value": [32, 32]}
        if net_struct['shared'] is not None:
            policy_only_layers = [net_struct['shared'][-1]] + net_struct['policy']
            value_only_layers = [net_struct['shared'][-1]] + net_struct['value']
            # ic(policy_only_layers, value_only_layers)
            for i in range(len(net_struct['shared'])-1):
                shared_net.append(nn.Linear(net_struct['shared'][i], net_struct['shared'][i+1]))
                shared_net.append(activation_fn())

        if net_struct['policy'] is not None:
            for i in range(len(policy_only_layers) - 1):
                policy_net.append(nn.Linear(policy_only_layers[i], policy_only_layers[i + 1]))
                policy_net.append(activation_fn())

        if net_struct['value'] is not None:
            for i in range(len(value_only_layers) - 1):
                value_net.append(nn.Linear(value_only_layers[i], value_only_layers[i + 1]))
                value_net.append(activation_fn())
            value_net.append(nn.Linear(value_only_layers[-1], 1))

        # input dimension for creating distributions
        self.latent_dim_policy = policy_only_layers[-1]

        # Create networks: if the list is empty, sequential model equals identity map
        self.shared_net = nn.Sequential(*shared_net).to(device)
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent)


# @dataclass
# class EqlNetworkConfig(NetworkConfig):
#     eql_config: EQLConfig = None
#     value_net_struct: list = None
#     value_activation: Type[nn.Module] = None
#
#
# class EQLNetwork(nn.Module):
#     def __init__(self, c: EqlNetworkConfig):
#         super(EQLNetwork, self).__init__()
#         device = c.device
#         s_dim = get_space_dim(c.env.observation_space)
#         ic(s_dim)
#         c.eql_config.in_feats = s_dim
#
#         self.latent_dim_policy = c.eql_config.structs[-1].out_dim()
#         self.policy_net = EQL(c.eql_config, EQLTrain(), device=device).to(device)
#         # self.policy_net.print(info=True)
#
#         value_net = []
#         c.value_net_struct.insert(0, s_dim)
#         for i in range(len(c.value_net_struct) - 1):
#             value_net.append(nn.Linear(c.value_net_struct[i], c.value_net_struct[i + 1]))
#             value_net.append(c.value_activation())
#         value_net.append(nn.Linear(c.value_net_struct[-1], 1))
#         self.value_net = nn.Sequential(*value_net).to(device)
#
#     def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         :return: latent_policy, latent_value of the specified network.
#             If all layers are shared, then ``latent_policy == latent_value``
#         """
#         features = torch.unsqueeze(features, 0)
#         # ic(features)
#         action, x = self.policy_net(features)
#         if isinstance(action, tuple):
#             action, pen_div = action[0], action[1]
#             # if div:
#             #     optimizer_div.zero_grad(set_to_none=True)
#             #     pen_div.backward(retain_graph=True)
#         action = torch.squeeze(action)
#         # ic(action, action.shape)
#         return action, self.value_net(features)


# def network_factory(network_config: NetworkConfig = None):
#     if isinstance(network_config, MlpNetworkConfig):
#         return MlpNetwork(network_config)
#     # elif isinstance(network_config, EqlNetworkConfig):
#     #     return EQLNetwork(network_config)
#     else:
#         raise NotImplementedError(f"Network type {network_config.__class__} is not supported")

def network_factory(network_type: str = None, network_config: Union[Dict, None] = None, device: torch.device = None):
    networks = {
        'mlp': MlpNetwork,
    }
    return networks[network_type](network_config, device)
