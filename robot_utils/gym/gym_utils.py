import logging
import numpy as np
import torch
import torch.nn.functional as F
import gym
from gym import spaces


def get_space_dim(space: spaces.Space) -> int:
    """
    Get the space dimension based on the gym Space instance.
    """
    if isinstance(space, spaces.Box):
        return int(np.prod(space.shape))
    elif isinstance(space, spaces.Discrete):
        return space.n
    elif isinstance(space, spaces.MultiDiscrete):
        return int(len(space.nvec))
    elif isinstance(space, spaces.MultiBinary):
        return int(space.n)
    else:
        raise NotImplementedError("space {} is not supported".format(space))


def get_space_dims(env: gym.Env):
    """
    return dimension of action space and observation space
    """
    return get_space_dim(env.action_space), get_space_dim(env.observation_space)


def get_action_bound(env: gym.Env):
    """
    return the action boundary [low, high] if action space is of type Box
    """
    if isinstance(env.action_space, spaces.Box):
        return env.action_space.low, env.action_space.high
    else:
        return None, None


def is_image_space_channels_first(observation_space: spaces.Box) -> bool:
    """
    Check if an image observation space (see ``is_image_space``)
    is channels-first (CxHxW, True) or channels-last (HxWxC, False).

    Use a heuristic that channel dimension is the smallest of the three.
    If second dimension is smallest, raise an exception (no support).

    :param observation_space:
    :return: True if observation space is channels-first image, False if channels-last.
    """
    smallest_dimension = np.argmin(observation_space.shape).item()
    if smallest_dimension == 1:
        logging.warning("Treating image space as channels-last, while second dimension was smallest of the three.")
    return smallest_dimension == 0


def is_image_space(observation_space: spaces.Space, channels_last: bool = True, check_channels: bool = False) -> bool:
    """
    Check if a observation space has the shape, limits and dtype of a valid image.
    The check is conservative, so that it returns False if there is a doubt.

    Valid images: RGB, RGBD, GrayScale with values in [0, 255]
    Args:
        observation_space:
        channels_last:
        check_channels: Whether to do or not the check for the number of channels.
        e.g., with frame-stacking, the observation space may have more channels than expected.

    Returns:

    """
    if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3:
        # Check the type
        if observation_space.dtype != np.uint8:
            return False

        # Check the value range
        if np.any(observation_space.low != 0) or np.any(observation_space.high != 255):
            return False

        # Skip channels check
        if not check_channels:
            return True
        # Check the number of channels
        if channels_last:
            n_channels = observation_space.shape[-1]
        else:
            n_channels = observation_space.shape[0]
        # RGB, RGBD, GrayScale
        return n_channels in [1, 3, 4]
    return False


def preprocess_obs(obs: torch.Tensor, observation_space: spaces.Space, normalize_images: bool = True) -> torch.Tensor:
    """
    Preprocess the observation:
    - For images, it normalizes pixel values to range [0, 1] by dividing 255.
    - For discrete observations, it creates a one hot vector.
    - For multi-dim discrete case, it concatenates of one hot encodings of each Categorical sub-space.
    - For binary case, it convert boolean to float.
    """
    if isinstance(observation_space, spaces.Box):
        if is_image_space(observation_space) and normalize_images:
            return obs.float() / 255.0
        return obs.float()

    elif isinstance(observation_space, spaces.Discrete):
        return F.one_hot(obs.long(), num_classes=observation_space.n).float()

    elif isinstance(observation_space, spaces.MultiDiscrete):
        return torch.cat(
            [
                F.one_hot(obs_.long(), num_classes=int(observation_space.nvec[idx])).float()
                for idx, obs_ in enumerate(torch.split(obs.long(), 1, dim=1))
            ],
            dim=-1,
        ).view(obs.shape[0], sum(observation_space.nvec))

    elif isinstance(observation_space, spaces.MultiBinary):
        return obs.float()

    else:
        raise NotImplementedError(f"Preprocessing not implemented for {observation_space}")


def get_flattened_obs_dim(observation_space: spaces.Space) -> int:
    """
    Get the dimension of the observation space when flattened.
    It does not apply to image observation space.

    Used by the ``FlattenExtractor`` to compute the input shape.

    :param observation_space:
    :return:
    """
    # See issue https://github.com/openai/gym/issues/1915
    # it may be a problem for Dict/Tuple spaces too...
    if isinstance(observation_space, spaces.MultiDiscrete):
        return sum(observation_space.nvec)
    else:
        # Use Gym internal method
        return spaces.utils.flatdim(observation_space)