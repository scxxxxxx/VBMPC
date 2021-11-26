import gym
import logging
import warnings
import numpy as np
from typing import Union, Optional, Dict, Callable, Type
from gym import spaces

import torch
from torch import nn
from torch.nn import functional as F
from robot_utils.gym.gym_utils import is_image_space, get_flattened_obs_dim, preprocess_obs


class BaseFeaturesExtractor(nn.Module):
    """
    Base class that represents a features extractor.

    :param observation_space:
    :param features_dim: Number of features extracted.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 0, **kwargs):
        super(BaseFeaturesExtractor, self).__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class FeatureExtractorFactory:
    """ The policy factory class"""

    registry = {}  # type: Dict[str, Type[BaseFeaturesExtractor]]
    """ Internal registry for available policies """

    @classmethod
    def register(cls, name: str) -> Callable:
        """ Class method to register sub-class to the internal registry.
        Args:
            name (str): The name of the sub-class.
        Returns:
            The sub-class itself.
        """

        def inner_wrapper(wrapped_class: Type[BaseFeaturesExtractor]) -> Type[BaseFeaturesExtractor]:
            if name in cls.registry:
                logger.warning(f'Policy {name} already exists. Will replace it.')
            cls.registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper


    @classmethod
    def create_model(cls,
                     name:          str, observation_space: gym.Space,
                     features_dim:  Optional[int],
                     config:        Optional[Dict] = None
                     ) -> Type[BaseFeaturesExtractor]:
        """ Factory method to create the policy with the given name and pass parameters as``kwargs``.
        Args:
            name (str): The name of the policy to create.
        Returns:
            An instance of the policy that is created.
        """
        if name is None:
            return None
        if config is None:
            config = {}
        logging.info(f"available policy factory {cls.registry}")
        logging.info(f"create policy with name {name}")
        if name not in cls.registry:
            logging.warning(f'Policy {name} does not exist in the registry')
            return None

        model_class = cls.registry[name]
        model = model_class(observation_space, features_dim, **config)
        return model


@FeatureExtractorFactory.register("FlattenExtractor")
class FlattenExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space):
        super(FlattenExtractor, self).__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.flatten(observations)


@FeatureExtractorFactory.register("NatureCNN")
class NatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(NatureCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


# def get_feature_extractor(feature_extractor_type: str, observation_space: gym.spaces.Box, config: Optional[Dict]):
#     feature_extractors = {
#         "flatten": FlattenExtractor,
#         'nature_cnn': NatureCNN
#     }
#     if feature_extractor_type in feature_extractors:
#         return feature_extractors[feature_extractor_type](observation_space, **config)
#     else:
#         raise RuntimeError(f"feature_extractor type {feature_extractor_type} is not supported")