import os
import logging
from abc import ABC, abstractmethod
from marshmallow_dataclass import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable
import gym
import numpy as np
import torch
from torch import nn

from robot_policy.rl.common.base_distribution import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    distribution_factory,
)
from robot_policy.rl.common.base_network import network_factory
from robot_policy.rl.common.base_feature_extraction import FeatureExtractorFactory
from robot_utils.torch.torch_utils import create_mlp, get_optimizer, get_scheduler
from robot_utils.py.utils import create_path, load_dataclass
from robot_utils.gym.gym_utils import is_image_space, preprocess_obs

from stable_baselines3.common.utils import is_vectorized_observation
from stable_baselines3.common.vec_env import VecTransposeImage


@dataclass
class PolicyConfig:
    policy_type:                str = None

    # model config
    model_type:                 Optional[str] = None
    model_config:               Optional[Dict[str, Any]] = None
    a_dim:                      Union[int, Tuple[int]] = 0  # TODO: check whether the type is correct for image space
    s_dim:                      Union[int, Tuple[int]] = 0
    a_min:                      Union[List[Any], None] = None  # TODO: serialize NumpyArray with marshmallow_numpy?
    a_max:                      Union[List[Any], None] = None
    squash_output:              Optional[bool] = None

    # optimizer & scheduler
    optimizer_type:             Optional[str] = None
    optimizer_config:           Optional[Dict[str, Any]] = None
    scheduler_type:             Optional[str] = None
    scheduler_config:           Optional[Dict[str, Any]] = None

    # feature extractor
    features_extractor_type:    Optional[str] = None
    features_extractor_config:  Optional[Dict[str, Any]] = None
    normalize_images:           Optional[bool] = None

    @staticmethod
    def _check_dict_empty(dic):
        return dic if dic else {}

    def __post_init__(self):
        self.model_config = self._check_dict_empty(self.model_config)
        self.optimizer_config = self._check_dict_empty(self.optimizer_config)
        self.scheduler_config = self._check_dict_empty(self.scheduler_config)
        self.features_extractor_config = self._check_dict_empty(self.features_extractor_config)


class BasePolicy(nn.Module, ABC):
    """
    The base model object: makes predictions in response to observations.

    In the case of policies, the prediction is an action. In the case of critics, it is the
    estimated value of the observation.
    """
    def __init__(self, env: gym.Env, config: Dict, model_path: str, device: torch.device, **kwargs):
        super(BasePolicy, self).__init__()
        # Note: configuration
        self.model_path = create_path("./saved_model/base_policy" if model_path is None else model_path)
        # self.policy_config_file = os.path.join(self.model_path, f"{self.name}_policy_config.pt")
        self.policy_params_file = os.path.join(self.model_path, f"{self.name}_policy_params.pt")
        self._load_config(config)
        # torch.save(self.c, self.policy_config_file)

        self.env = env
        self.device = device

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self._init_feature_extractor()
        self._init_policy_networks()
        self._init_optimizer()

    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def _load_config(self, config: Dict):
        """Implement your own method to load configuration"""
        self.c = load_dataclass(PolicyConfig, config)

    @abstractmethod
    def _init_policy_networks(self):
        raise NotImplementedError

    def _init_feature_extractor(self):
        """
        initialize feature extractor, which extracts hidden features from the raw input.
        """
        self.normalize_images = self.c.normalize_images
        self.features_extractor = FeatureExtractorFactory.create_model(
            self.c.features_extractor_type, self.observation_space, self.c.features_extractor_config
        )

    def _init_optimizer(self):
        self.optimizer = get_optimizer(self.c.optimizer_type, self.parameters(), self.c.optimizer_config)
        self.scheduler = get_scheduler(self.c.scheduler_type, self.optimizer, self.c.scheduler_config)

    @abstractmethod
    def forward(self, *args, **kwargs) -> Union[np.ndarray, List]:
        del args, kwargs

    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Preprocess the observation if needed and extract features.
        """
        assert self.features_extractor is not None, "No features extractor was set"
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return self.features_extractor(preprocessed_obs)

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    @abstractmethod
    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.
        """

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the action and state from policy based on the observation.
        Args:
            observation:
            state: The last states (can be None, used in recurrent policies)
            mask: The last masks (can be None, used in recurrent policies)
            deterministic: flag to control deterministic or stochastic actions

        Returns: the model's action and the next state (can be used in recurrent policy)
        """
        # COMM: handle observation
        # TODO (GH/1): add support for RNN policies
        # if state is None:
        #     state = self.initial_state
        # if mask is None:
        #     mask = [False for _ in range(self.n_envs)]
        # if isinstance(observation, dict):
        #     observation = ObsDictWrapper.convert_dict(observation)
        # else:
        # observation = np.array(observation)

        # Handle the different cases for images
        # as PyTorch use channel first format
        if is_image_space(self.observation_space):
            if not (
                observation.shape == self.observation_space.shape or observation.shape[1:] == self.observation_space.shape
            ):
                # Try to re-order the channels
                transpose_obs = VecTransposeImage.transpose_image(observation)
                if (
                    transpose_obs.shape == self.observation_space.shape
                    or transpose_obs.shape[1:] == self.observation_space.shape
                ):
                    observation = transpose_obs

        vectorized_env = is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)

        observation = torch.as_tensor(observation).float().to(self.device)

        with torch.no_grad():
            actions = self._predict(observation, deterministic=deterministic)
        # convert to numpy
        actions = actions.cpu().numpy()

        # handle action
        actions = self.unscale_action(actions)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            # actions = actions[0]
            actions = actions.squeeze()

        return actions, state

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        """
        if self.c.squash_output and isinstance(self.action_space, gym.spaces.Box):
            low, high = self.action_space.low, self.action_space.high
            return 2.0 * ((action - low) / (high - low)) - 1.0
        else:
            return action

    def unscale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high] and make sure it's clipped to the valid interval
        """
        if isinstance(self.action_space, gym.spaces.Box):
            if self.c.squash_output:
                low, high = self.action_space.low, self.action_space.high
                action = low + (0.5 * (action + 1.0) * (high - low))
            return np.clip(action, self.action_space.low, self.action_space.high)
        else:
            return action

    def save_model(self, best_param=None):
        """
        Save the policy parameter and configuration. You can also pass the best param from outside.
        """
        param = best_param if best_param else self.state_dict()
        torch.save(param, self.policy_params_file)
        # torch.save(self.c, self.policy_config_file)

    def load_model(self):
        """
        Load the model parameter
        """
        self.load_state_dict(torch.load(self.policy_params_file))


class PolicyFactory:
    """ The policy factory class """
    registry = {}  # type: Dict[str, Type[BasePolicy]]

    """ Internal registry for available policies """
    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Class method to register sub-class to the internal registry.
        Args:
            name (str): The name of the sub-class.
        Returns:
            The sub-class itself.
        """
        def inner_wrapper(wrapped_class: Type[BasePolicy]) -> Type[BasePolicy]:
            if name in cls.registry:
                logging.warning(f'Policy {name} already exists. Will replace it.')
            cls.registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @classmethod
    def create_model(cls, name: str) -> Union[Type[BasePolicy], None]:
        """
        Factory method to create the policy with the given name and pass parameters as``kwargs``.
        Args:
            name: The name of the policy to create.

        Returns: An instance of the policy that is created.
        """
        # logging.info(f"available policy factory {cls.registry}")
        logging.info(f"create policy with name: {name}")
        if name not in cls.registry:
            logging.warning(f'Policy {name} does not exist in the registry')
            return None

        model_class = cls.registry[name]
        return model_class


@dataclass
class ActorCriticPolicyConfig(PolicyConfig):
    log_std_init: float = 0.0
    ortho_init: bool = True


@PolicyFactory.register("ActorCriticSimplePolicy")
class ActorCriticSimplePolicy(BasePolicy):
    def __init__(self,
                 env: gym.Env,
                 config: Optional[Dict],
                 model_path: str,
                 device: torch.device):
        super(ActorCriticSimplePolicy, self).__init__(env, config, model_path, device)

    def _load_config(self, config: Dict):
        """Implement your own method to load configuration"""
        self.c = load_dataclass(ActorCriticPolicyConfig, config)

    def _init_policy_networks(self) -> None:
        """
        Create the networks for shared_net, action_net and value_net
        """
        self.c.model_config.update({
            "a_dim": self.c.a_dim,
            "s_dim": self.c.s_dim
        })
        self.actor_critic = network_factory(self.c.model_type, self.c.model_config, self.device)
        self.action_dist = distribution_factory(self.action_space, dist_kwargs=None)
        self._build_action_net()

        if self.c.ortho_init:
            module_gains = {
                # self.features_extractor: np.sqrt(2),
                self.actor_critic: np.sqrt(2),
                self.action_net: 0.01,
                # self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

    def _build_action_net(self):
        """
        Based on the latent dimension and distribution type, construct the action output network.
        """
        # get input dimension of distribution network
        latent_action_dim = self.actor_critic.latent_dim_policy

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.distribution_net(
                latent_dim=latent_action_dim, log_std_init=self.c.log_std_init
            )
        elif isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = self.action_dist.distribution_net(latent_dim=latent_action_dim)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            self.action_net = self.action_dist.distribution_net(latent_dim=latent_action_dim)
        elif isinstance(self.action_dist, BernoulliDistribution):
            self.action_net = self.action_dist.distribution_net(latent_dim=latent_action_dim)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic), return the action, values, and the log_prob(action)
        """
        latent_pi, values = self._get_latent(obs.float())
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.logp(actions)

        return actions, values, log_prob

    def _get_latent(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """get latent representation of actions to build up the action distribution."""
        latent_pi, latent_vf = self.actor_critic(obs.float())
        return latent_pi, latent_vf

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        """
        Given the latent features of action (latent_pi), get the action distribution in latent space
        """
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.prob_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.prob_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.prob_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.prob_distribution(action_logits=mean_actions)
        else:
            raise ValueError("Invalid action distribution")

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Given observation, get the next (default: deterministic) action.
        """
        latent_pi, _ = self._get_latent(observation)
        distribution = self._get_action_dist_from_latent(latent_pi)
        return distribution.get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given observation, get the values, log probability and entropy of the action distribution.
        """
        latent_pi, values = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.logp(actions)
        return values, log_prob, distribution.entropy()


def create_sde_features_extractor(
        features_dim: int, sde_struct: List[int], activation_fn: nn.Module
) -> Tuple[nn.Sequential, int]:
    """

    Args:
        features_dim: input dimension
        sde_struct:
        activation_fn:

    Returns:

    """
    # make sure there is at least one hidden layer
    sde_struct.insert(0, features_dim)
    sde_activation = activation_fn if len(sde_struct) > 1 else None
    latent_sde_dim = sde_struct[-1] if len(sde_struct) > 1 else features_dim
    sde_features_extractor = create_mlp(sde_struct, activation_fn=sde_activation, squash_output=False)
    return sde_features_extractor, latent_sde_dim



