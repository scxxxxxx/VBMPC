import os
import jax
import jax.numpy as jnp
import numpy as np
import glob
from robot_utils.py.utils import create_path


def save_param(path, parameters, best=False):
    if best:
        path += "_best"
    create_path(path)
    if len(os.listdir(path)) > 0:
        for f in glob.glob("param_[0-9]*.npy"):
            os.remove(f)
    leaves, _ = jax.tree_flatten(parameters)
    for leaf_idx, parameter in enumerate(leaves):
        file = os.path.join(path, "param_{}.npy".format(leaf_idx))
        jnp.save(file, parameter)


def load_param(path, treedef):
    leaves = []
    for leaf_idx in range(len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])):
        file = os.path.join(path, "param_{}.npy".format(leaf_idx))
        parameter = jnp.load(file)
        leaves.append(parameter)
    return jax.tree_unflatten(treedef, leaves)


@jax.jit
def nmse(target, pred):
    # (target - pred)/target = 1 - pred/target
    ratio = pred / target
    return mse(1., ratio)


@jax.jit
def mse(target, pred):
    return jnp.mean((target - pred) ** 2)


def wrap_coords(state):
    # wrap generalized coordinates to [-pi, pi]
    return jnp.concatenate([(state[:2] + jnp.pi) % (2 * jnp.pi) - jnp.pi, state[2:]])


def wrap_coords_numpy(state):
    # wrap generalized coordinates to [-pi, pi]
    return np.concatenate([(state[:2] + np.pi) % (2 * np.pi) - np.pi, state[2:]])
