import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.nn.initializers import glorot_normal, normal, glorot_uniform


def DivisionLayer(out_dim, W_init=glorot_normal(), b_init=normal()):
    lin_out_dim = 2 * out_dim

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = random.split(rng)
        W, b = W_init(k1, (input_shape[-1], lin_out_dim)), b_init(k2, (lin_out_dim,))
        return output_shape, (W, b)

    def apply_fun(params, inputs, **kwargs):
        theta = kwargs.get("theta", 0.5)
        W, b = params
        x = jnp.dot(inputs, W) + b
        y1, y2 = jnp.split(x, [out_dim])
        res = (y2 > theta) * jnp.divide(y1, y2)
        return res, y2
    return init_fun, apply_fun


def OperatorLayer(struct, W_init=glorot_normal(), b_init=normal()):
    # struct [n_identity, n_sin, n_cos, n_multiply]
    struct = jnp.array(struct)
    lin_out_dim = jnp.sum(struct[:-1]) + 2 * struct[-1]
    out_dim = jnp.sum(struct)
    split_idx = jax.lax.cumsum(struct)
    split_idx = np.array(split_idx)

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = random.split(rng)
        W, b = W_init(k1, (input_shape[-1], lin_out_dim)), b_init(k2, (lin_out_dim,))
        return output_shape, (W, b)

    def apply_fun(params, inputs, **kwargs):
        W, b = params
        x = jnp.dot(inputs, W) + b
        y1, y2, y3, y4, y5 = jnp.split(x, split_idx)
        return jnp.concatenate((y1, jnp.sin(y2), jnp.cos(y3), y4 * y5))
    return init_fun, apply_fun
