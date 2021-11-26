import copy
import os
import random
import coloredlogs
import logging
import numpy as np
import gym

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataclasses import field
from marshmallow import validate
from typing import Any, Union, Type, List, Sized, Dict, Iterator
from icecream import install, ic
from torch.utils.data import Sampler
install()
ic.configureOutput(includeContext=True)
coloredlogs.install(fmt='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s | %(lineno)d] %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d:%H:%M:%S')


def get_device(use_gpu: bool = True):
    cuda_available = torch.cuda.is_available()
    return torch.device("cuda:0" if cuda_available and use_gpu else "cpu")


def init_torch(seed=None, use_gpu=False, default_type=torch.float32):
    device = get_device(use_gpu)
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available() and use_gpu:
            torch.cuda.manual_seed(seed)

    torch.set_default_dtype(default_type)
    return device


def zero_loss(device):
    # return Variable(torch.FloatTensor([0]).to(device))
    return Variable(torch.tensor([0.0], device=device).float())


def get_all_data(dataset):
    dataloader = DataLoader(dataset, batch_size=len(dataset), num_workers=4, pin_memory=True)
    return next(iter(dataloader))


# def gradient(y, x):
#     return torch.autograd.grad(y, x, create_graph=True)[0]
def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def hessian_(y, x0, x1):
    grad = gradient(y, x0)  # (B, f_x0)
    hess = torch.zeros(y.shape[0], x0.shape[-1], x1.shape[-1]).to(y.device)
    for k in range(grad.shape[1]):
        hess[..., k, :] = gradient(grad[:, k].reshape(-1, 1), x1)
    return hess


def hessian(y, x0, x1_list):
    grad = gradient(y, x0)  # (B, f_x0)
    hess_list = []
    for x1 in x1_list:
        hess = torch.zeros(x0.shape[0], x0.shape[-1], x1.shape[-1]).to(y.device)
        for k in range(grad.shape[1]):
            hess[..., k, :] = gradient(grad[:, k].reshape(-1, 1).sum(), x1)
        hess_list.append(hess)
    return (grad, *hess_list)


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def get_mesh_grid(side_length, dim=2, range=None):
    """
    Generates a flattened grid of (x,y,...) coordinates
    Args:
        side_length: int or list/tuple of ints. int, generate same number of samples for each dim.
                     list, generate different number of samples for each dim
        dim: when side_length is int, you need to specify dimension of the coordinates
        range: a list of tuple, [(min, max), (min, max) ... ] specify the sample range of each dim

    Returns: flattened grid as 2D matrix, each row is a sampled coordinates

    """
    # tensors = tuple(dim * [torch.linspace(-1, 1, steps=side_length, dtype=torch.float64)])
    if isinstance(side_length, int):
        if range is None:
            tensors = tuple(dim * [torch.linspace(-1, 1, steps=side_length)])
        else:
            tensors = tuple(dim * [torch.linspace(range[0], range[1], steps=side_length)])
    else:
        if range is None:
            tensors = tuple([torch.linspace(-1, 1, steps=s) for s in side_length])
        else:
            tensors = tuple([torch.linspace(r[0], r[1], steps=s) for s, r in zip(side_length, range)])
    mesh_grid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mesh_grid = mesh_grid.reshape(-1, dim)
    return mesh_grid


def get_image_mesh_grid(height, width):
    x = torch.arange(width)
    y = torch.arange(height)
    mesh = torch.meshgrid(x, y)
    mesh_grid = torch.stack((mesh[0].transpose(1, 0), mesh[1].transpose(1, 0)), dim=-1)
    mesh_grid = mesh_grid.reshape(-1, 2)
    return mesh_grid


def get_2d_mesh(l1, l2=None, range=None):
    """
    
    Args:
        l1 (int): the sparse number of points
        l2 (int): the dense number of points, default l2 = 10 * l1
        range (list(list(float)):  [[x_min, x_max], [y_min, y_max]]

    Returns:

    """
    if l2 is None:
        l2 = 10 * l1
    side_length = (l1, l2)
    tensors1 = tuple([torch.linspace(r[0], r[1], steps=s) for s, r in zip(side_length, range)])
    mesh_grid1 = torch.stack(torch.meshgrid(*tensors1), dim=-1)
    mesh_grid1 = mesh_grid1.reshape(-1, 2)

    side_length = (l2, l1)
    tensors2 = tuple([torch.linspace(r[0], r[1], steps=s) for s, r in zip(side_length, range)])
    mesh_grid2 = torch.stack(torch.meshgrid(*tensors2), dim=-1)
    mesh_grid2 = mesh_grid2.reshape(-1, 2)

    mesh_grid = torch.cat((mesh_grid1, mesh_grid2), dim=0)
    return mesh_grid


def tanh(x, a=1.0):
    y = torch.exp(a * x)
    return (y - 1.0/y) / (y + 1.0/y)


def sigmoid(x, center=0.0, alpha=1.0):
    return 1.0 / (1 + torch.exp(- alpha * (x - center)))


def prepare_data(dataset, train_data_ratio=0.8, **kwargs):
    train_size = int(train_data_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_data, val_data


def split_indices(dataset, train_data_ratio=1.0, **kwargs):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(np.floor(train_data_ratio * dataset_size))
    return indices[:split], indices[split:]


class ChunkSequentialBatchSampler(Sampler[List[int]]):

    def __init__(self, chunk: List[int], batch_size: int, drop_last: bool) -> None:
        self.chunk = chunk
        self.all_idx = range(sum(self.chunk))
        self.chunk_idx = 0

        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.all_idx:
            if idx == self.chunk[self.chunk_idx] and len(batch) > 0:
                yield batch
                batch = []
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        return len(self.chunk)


class TrajSequentialBatchSampler(Sampler[List[int]]):

    def __init__(self, chunk: List[int], batch_size: int, drop_last: bool) -> None:
        self.chunk = chunk
        self.all_idx = range(sum(self.chunk))
        self.chunk_idx = 0
        self.chunk_cumsum = np.cumsum(chunk)
        self.chunk_start_idx = [0] + list(self.chunk_cumsum[:-1])
        self.batchsize_per_chunk = batch_size // len(chunk)
        if self.batchsize_per_chunk < 1:
            raise RuntimeError(f"batch size have to be larger than {len(chunk)}")

        self.max_chunk_len = max(chunk)
        if self.max_chunk_len % self.batchsize_per_chunk == 0:
            self.num_batches = self.max_chunk_len // self.batchsize_per_chunk
        else:
            self.num_batches = self.max_chunk_len // self.batchsize_per_chunk + 1

        self.batch_size = self.batchsize_per_chunk * len(chunk)
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in range(self.num_batches):
            for chunk_idx, chunk_start_idx in enumerate(list(self.chunk_start_idx)):
                idx_in_current_chunk = np.arange(self.batchsize_per_chunk) + chunk_start_idx + idx * self.batchsize_per_chunk
                idx_in_current_chunk_clip = np.clip(idx_in_current_chunk, idx_in_current_chunk.min(), self.chunk_cumsum[chunk_idx]-1)
                # ic(idx_in_current_chunk, idx_in_current_chunk_clip)
                batch += list(idx_in_current_chunk_clip)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        return len(self.chunk)


def create_mlp(net_struct: List[int], activation_fn: torch.nn.Module = torch.nn.ReLU, squash_output: bool = False
) -> torch.nn.Sequential:
    """
    Construct a sequential model of multi-layer perceptron. If only input_dim is given,  (net_struct = [input_dim, ],
    then it returns an identity map.
    Args:
        net_struct: [input_dim, hidden_dim_1, ...,  hidden_dim_n, out_dim]
        activation_fn:
        squash_output:

    Returns: torch sequential model

    """
    modules = []
    for idx in range(len(net_struct) - 1):
        modules.append(torch.nn.Linear(net_struct[idx], net_struct[idx + 1]))
        modules.append(activation_fn())

    if squash_output:
        modules.append(torch.nn.Tanh())
    return torch.nn.Sequential(*modules)


class AdaptiveLinear(torch.nn.Linear):
    def __init__(self, in_feats, out_feats, bias=True, aa_type=None, adaptive_rate_scale=None):
        super(AdaptiveLinear, self).__init__(in_features=in_feats, out_features=out_feats, bias=bias)

        self.adaptive_rate_scale = adaptive_rate_scale if adaptive_rate_scale else 10.0
        if aa_type == 'layer':
            self.scale = torch.nn.Parameter(1.0 / self.adaptive_rate_scale * torch.ones(1))
        elif aa_type == 'neuron':
            if self.adaptive_rate_scale:
                self.scale = torch.nn.Parameter(1.0 / self.adaptive_rate_scale * torch.ones(self.in_features))
        else:
            self.adaptive_rate_scale = None

    def forward(self, x: torch.Tensor):
        if self.adaptive_rate_scale:
            return torch.nn.functional.linear(self.adaptive_rate_scale * self.scale * x, self.weight, self.bias)
        return torch.nn.functional.linear(x, self.weight, self.bias)


def create_linear_block(in_feats, out_feats, activation: str, dropout_rate=None, aa_type=None, adaptive_rate_scale=None):
    if dropout_rate is None:
        return torch.nn.Sequential(
            AdaptiveLinear(in_feats, out_feats, aa_type=aa_type, adaptive_rate_scale=adaptive_rate_scale),
            getattr(torch.nn, activation)(),
        )
    else:
        return torch.nn.Sequential(
            AdaptiveLinear(in_feats, out_feats, aa_type=aa_type, adaptive_rate_scale=adaptive_rate_scale),
            getattr(torch.nn, activation)(),
            torch.nn.Dropout(dropout_rate),
        )


def create_sequential_adaptive_linear(
        structure: list,
        activation: str,
        dropout_rate: Union[float, None] = None,
        adaptive_type: str = field(metadata={"validate": validate.OneOf(["global", "layer", "neuron"])}),
        scaling: Union[float, None] = None,
        last_layer_linear: bool = True
):
    if last_layer_linear:
        return torch.nn.Sequential(
            *[create_linear_block(in_feats, out_feats, activation, dropout_rate, adaptive_type, scaling)
              for in_feats, out_feats in zip(structure[:-2], structure[1:-1])
              ],
            AdaptiveLinear(structure[-2], structure[-1])
        )
    else:
        return torch.nn.Sequential(
            *[create_linear_block(in_feats, out_feats, activation, dropout_rate, adaptive_type, scaling)
              for in_feats, out_feats in zip(structure[:-1], structure[1:])
              ]
        )


class LinearWithNoneNegativeWeights(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearWithNoneNegativeWeights, self).__init__()
        self.weight = torch.nn.parameter.Parameter(torch.Tensor(out_dim, in_dim))

    def forward(self, input):
        ## softplus does not work, why???
        return torch.nn.functional.linear(input, torch.nn.functional.relu(self.weight), bias=None)



class RandomSequentialSampler(torch.utils.data.Sampler[int]):
    def __init__(self, data_source: Sized):
        # super(RandomSequentialSampler, self).__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        idx = np.arange(len(self.data_source))
        np.random.shuffle(idx)
        return iter(idx)

    def __len__(self) -> int:
        return len(self.data_source)


def set_random_seeds(random_seed, use_gpu=True):
    # python env
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    # python RNG
    random.seed(random_seed)
    # numpy RNG
    np.random.seed(random_seed)
    # torch RNG for GPU and CPU device
    torch.manual_seed(random_seed)
    if use_gpu and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(random_seed)
        torch.cuda.manual_seed(random_seed)
    # gym RNG
    if hasattr(gym.spaces, 'prng'):
        gym.spaces.prng.seed(random_seed)


def tensor_clamp(value: torch.Tensor, value_min: torch.Tensor, value_max: torch.Tensor):
    value = torch.where(value > value_min, value, value_min)
    value = torch.where(value < value_max, value, value_max)
    return value


def copy_grad(from_model, to_model, set_from_gradients_to_zero=False):
    for from_param, to_param in zip(from_model.parameters(), to_model.parameters()):
        to_param._grad = from_param.grad.clone()
        if set_from_gradients_to_zero:
            from_param._grad = None


def copy_param(from_model, to_model):
    to_model.load_state_dict(copy.deepcopy(from_model.state_dict()))


def get_optimizer(name: str, param: Iterator[torch.Tensor], config: Dict[str, Any]):
    return getattr(torch.optim, name)(param, **config) if name else None


def get_scheduler(name: str, optimizer: Type[torch.optim.Optimizer], config: Dict):
    return getattr(torch.optim.lr_scheduler, name)(optimizer, **config) if name else None


def trace_of_jacobian(f, x):
    """calculate the trace of the Jacobian df/dx"""
    sum_diag = 0.
    for i in range(x.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), x, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        if isinstance(self.mean, torch.Tensor) and self.mean.dim() == 0:
            self.mean.unsqueeze(0)
            self.std.unsqueeze(0)

    def __call__(self, tensor, reverse: bool = False):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        if len(tensor.shape) == 1 or len(self.mean) == 1:
            if reverse:
                ic(self.std, self.mean, tensor.shape)
                tensor.mul_(self.std).add_(self.mean)
            else:
                tensor.sub_(self.mean).div_(self.std)
        elif tensor.shape[0] == len(self.mean) == len(self.std):
            for t, m, s in zip(tensor, self.mean, self.std):
                if reverse:
                    t.mul_(s).add_(m)
                else:
                    t.sub_(m).div_(s)
        elif tensor.shape[1] == len(self.mean) == len(self.std):
            if reverse:
                tensor.mul_(self.std).add_(self.mean)
            else:
                tensor.sub_(self.mean).div_(self.std)
        return tensor


def rand_index(pop_size, num_samples, device: str):
    vec = torch.unique(
        (torch.rand(num_samples, device=device) * pop_size).floor().long()
    )
    # Eliminate all duplicate entries. Might slow down the procedure but totally worth it.
    while vec.shape[0] != num_samples:
        vec = torch.unique(
            torch.cat(
                [
                    vec,
                    (torch.rand(num_samples - vec.shape[0], device=device) * pop_size)
                    .floor()
                    .long(),
                ]
            )
        )

    return vec.view(-1)


def rand_choice_gpu(pop_size, num_samples, shuffle=False):
    return torch.randperm(pop_size)[:num_samples] if shuffle else rand_index(pop_size, num_samples, device="cuda")