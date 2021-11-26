from abc import ABC, abstractmethod
from typing import Generator, Optional, List, Dict, Any
from dataclasses import dataclass
import numpy as np
import torch
import random


@dataclass
class RolloutSamples:
    obs: torch.Tensor
    act: torch.Tensor
    val: torch.Tensor
    logp: torch.Tensor
    adv: torch.Tensor
    ret: torch.Tensor


@dataclass
class ReplaySamples:
    obs: torch.Tensor
    act: torch.Tensor
    rew: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor


@dataclass
class ReplaySamplesExt:
    obs: torch.Tensor
    act: torch.Tensor
    rew: torch.Tensor
    next_obs: torch.Tensor
    next_act: torch.Tensor
    done: torch.Tensor


class Buffer(ABC):
    def __init__(
        self,
        capacity: int,
        a_dim: int,
        s_dim: int,
        device: torch.device,
        n_envs: int = 1,
    ):
        super(Buffer, self).__init__()
        self.capacity = capacity
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.idx = 0
        self.full = False
        self.device = device
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        return self.capacity if self.full else self.idx

    def reset(self) -> None:
        self.idx = 0
        self.full = False

    def add(self, *args, **kwargs):
        raise NotImplementedError()

    def extend(self, *args):
        for data in zip(*args):
            self.add(*data)

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default (disable it by setting "copy" to False)
        """
        if copy:
            return torch.tensor(array).to(self.device)
        return torch.as_tensor(array).to(self.device)


class RolloutBuffer(Buffer):
    def __init__(
            self,
            rollout_steps: int,
            a_dim: int,
            s_dim: int,
            device: torch.device,
            n_envs: int = 1,
            gae_lambda: float = 1,
            gamma: float = 0.99,
    ):
        super(RolloutBuffer, self).__init__(rollout_steps, a_dim, s_dim, device, n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.obs, self.act,  self.rew, self.adv = None, None, None, None
        self.ret, self.done, self.val, self.logp = None, None, None, None
        self.generator_ready = False
        self.reset()

    def reset(self):
        """ initialize the shape of rollout buffers """
        self.obs = np.zeros((self.capacity, self.n_envs) + (self.s_dim,), dtype=np.float32)
        self.act = np.zeros((self.capacity, self.n_envs, self.a_dim), dtype=np.float32)
        self.rew = np.zeros((self.capacity, self.n_envs), dtype=np.float32)
        self.ret = np.zeros((self.capacity, self.n_envs), dtype=np.float32)

        self.done = np.zeros((self.capacity, self.n_envs), dtype=np.float32)
        self.val  = np.zeros((self.capacity, self.n_envs), dtype=np.float32)
        self.logp = np.zeros((self.capacity, self.n_envs), dtype=np.float32)
        self.adv  = np.zeros((self.capacity, self.n_envs), dtype=np.float32)

        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def compute_returns_and_advantage(self, last_values: torch.Tensor, done: np.ndarray) -> None:
        """ compute the return value and the GAE """
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.capacity)):
            if step == self.capacity - 1:
                next_non_terminal = 1.0 - done
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.done[step + 1]
                next_values = self.val[step + 1]
            delta = self.rew[step] + self.gamma * next_values * next_non_terminal - self.val[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.adv[step] = last_gae_lam
        self.ret = self.adv + self.val

    def add(self, obs: np.ndarray, act: np.ndarray, rew: np.ndarray, done: np.ndarray,
            val: torch.Tensor, logp: torch.Tensor) -> None:
        """ add a new transition to the buffer """
        if len(logp.shape) == 0:
            # Reshape 0-d tensor to avoid error
            logp = logp.reshape(-1, 1)

        self.obs[self.idx] = np.array(obs).copy()
        self.act[self.idx] = np.array(act).copy()
        self.rew[self.idx] = np.array(rew).copy()
        self.done[self.idx] = np.array(done).copy()
        self.val[self.idx] = val.clone().cpu().numpy().flatten()
        self.logp[self.idx] = logp.clone().cpu().numpy()

        self.idx += 1
        self.full = (self.idx == self.capacity)

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutSamples, None, None]:
        """ get a generator containing 'batch_size' RolloutSamples """
        assert self.full, ""
        indices = np.random.permutation(self.capacity * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for tensor in ["obs", "act", "val", "logp", "adv", "ret"]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create mini batches
        if batch_size is None:
            batch_size = self.capacity * self.n_envs

        start_idx = 0
        while start_idx < self.capacity * self.n_envs:
            yield self._get_samples(indices[start_idx:start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_idx: np.ndarray) -> RolloutSamples:
        data = (
            self.obs[batch_idx],
            self.act[batch_idx],
            self.val[batch_idx].flatten(),
            self.logp[batch_idx].flatten(),
            self.adv[batch_idx].flatten(),
            self.ret[batch_idx].flatten(),
        )
        return RolloutSamples(*tuple(map(self.to_torch, data)))


class EpisodicReplayBuffer:
    '''
    Replay buffer containing a fixed maximun number of trajectories with
    the highest returns seen so far, designed for UDRL.
    '''

    def __init__(self, size=0):
        self.size = size
        self.buffer = []

    def add(self, episode):
        '''
        Params:
            episode (namedtuple):
                (states, actions, rewards, init_command, total_return, length)
        '''

        self.buffer.append(episode)

    def get(self, num):
        '''
        Params:
            num (int):
                get the last `num` episodes from the buffer
        '''

        return self.buffer[-num:]

    def random_batch(self, batch_size):
        '''
        Params:
            batch_size (int)

        Returns:
            Random batch of episodes from the buffer
        '''

        idxs = np.random.randint(0, len(self), batch_size)
        return [self.buffer[idx] for idx in idxs]

    def sort(self):
        '''Keep the buffer sorted in ascending order by total return'''

        key_sort = lambda episode: episode.total_return
        self.buffer = sorted(self.buffer, key=key_sort)[-self.size:]

    def save(self, filename):
        '''Save the buffer in numpy format

        Param:
            filename (str)
        '''

        np.save(filename, self.buffer)

    def load(self, filename):
        '''Load a numpy format file

        Params:
            filename (str)
        '''

        raw_buffer = np.load(filename)
        self.size = len(raw_buffer)
        self.buffer = \
            [make_episode(episode[0], episode[1], episode[2], episode[3], episode[4], episode[5]) \
             for episode in raw_buffer]

    def __len__(self):
        '''
        Returns:
            Size of the buffer
        '''
        return len(self.buffer)


class ReplayBuffer_MPPI:
    """Episodic replay buffer for MPPI"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, next_action, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, next_action, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, next_action, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, next_action, done

    def sample_all(self, ordered=False):
        if ordered:
            state, action, reward, next_state, next_action, done = map(np.stack, zip(*self.buffer))
            return state, action, reward, next_state, next_action, done
        else:
            return self.sample(self.position)

    def __len__(self):
        return len(self.buffer)


class ReplayBuffer(Buffer):
    def __init__(
            self,
            capacity: int,
            a_dim: int,
            s_dim: int,
            device: torch.device,
            sample_style: str = "normal",
            n_envs: int = 1,
    ):
        super(ReplayBuffer, self).__init__(capacity, a_dim, s_dim, device, n_envs)

        # TODO should we consider different data types of obs, act (e.g image int?)
        self.obs = np.zeros((self.capacity, self.n_envs) + (self.s_dim,), dtype=np.float32)
        self.act = np.zeros((self.capacity, self.n_envs, self.a_dim), dtype=np.float32)
        self.rew = np.zeros((self.capacity, self.n_envs), dtype=np.float32)
        self.done = np.zeros((self.capacity, self.n_envs), dtype=np.float32)
        self.timeout = np.zeros((self.capacity, self.n_envs), dtype=np.float32)

        if sample_style == "normal":
            self.sample = self.sample_normal
        elif sample_style == "extended":
            self.sample = self.sample_ext
        else:
            raise KeyError(f"style {sample_style} is not supported")

    def add(self, obs: np.ndarray, act: np.ndarray, rew: np.ndarray, next_obs: np.ndarray, done: np.ndarray,
            infos: List[Dict[str, Any]],) -> None:
        """ add transition to replay buffer """
        self.obs[self.idx] = np.array(obs).copy()
        self.obs[(self.idx + 1) % self.capacity] = np.array(next_obs).copy()
        self.act[self.idx] = np.array(act).copy()
        self.rew[self.idx] = np.array(rew).copy()
        self.done[self.idx] = np.array(done).copy()
        # self.timeout[self.idx] = np.array([info.get("TimeLimit.truncated", False) for info in infos])
        self.timeout[self.idx] = np.array(infos.get("TimeLimit.truncated", False))

        self.idx += 1
        if self.idx == self.capacity:
            self.full = True
            self.idx = 0

    def sample_ext(self, batch_size: int) -> ReplaySamplesExt:
        """ Sample elements from the replay buffer, also include next state and action """
        # select all indices that are not done or timed out
        max_idx = self.capacity if self.full else self.idx
        batch_idx = np.where((self.done[:max_idx] != 1.0) & (self.timeout[:max_idx] != 1.0))[0]
        batch_idx = np.random.choice(batch_idx, batch_size, replace=False)

        next_idx = (batch_idx + 1) % self.capacity
        next_obs = self.obs[next_idx, 0, :]
        next_act = self.act[next_idx, 0, :]
        data = (
            self.obs[batch_idx, 0, :],
            self.act[batch_idx, 0, :],
            self.rew[batch_idx],
            next_obs,
            next_act,
            self.done[batch_idx] * (1 - self.timeout[batch_idx]),
        )
        return ReplaySamplesExt(*tuple(map(self.to_torch, data)))

    def sample_normal(self, batch_size: int) -> ReplaySamples:
        """ Sample elements from the replay buffer """
        if self.full:
            batch_idx = (np.random.randint(1, self.capacity, size=batch_size) + self.idx) % self.capacity
        else:
            batch_idx = np.random.randint(0, self.idx, size=batch_size)

        next_idx = (batch_idx + 1) % self.capacity
        next_obs = self.obs[next_idx, 0, :]
        data = (
            self.obs[batch_idx, 0, :],
            self.act[batch_idx, 0, :],
            self.rew[batch_idx],
            next_obs,
            self.done[batch_idx] * (1 - self.timeout[batch_idx]),
        )
        return ReplaySamples(*tuple(map(self.to_torch, data)))

    def save(self, filename):
        np.save(filename, self.buffer)

    def load(self, filename):
        '''Load a numpy format file

        Params:
            filename (str)
        '''

        raw_buffer = np.load(filename)
        self.size = len(raw_buffer)
        self.buffer = \
            [make_episode(episode[0], episode[1], episode[2], episode[3], episode[4], episode[5]) \
             for episode in raw_buffer]
