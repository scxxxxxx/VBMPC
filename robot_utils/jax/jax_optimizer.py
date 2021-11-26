from jax.experimental.optimizers import *


def update_opt_state(parameters, opt_state):
    """

    Args:
        parameters: the model parameters
        opt_state: the optimizer state

    Returns:

    """
    states_flat, tree, subtrees = opt_state
    leaves, _ = tree_flatten(parameters)
    new_states = tuple([p, m0, v0] for p, (x0, m0, v0) in zip(leaves, states_flat))
    new_states_flat, _ = unzip2(map(tree_flatten, new_states))
    return OptimizerState(new_states_flat, tree, subtrees)


@optimizer
def adam(step_size, b1=0.9, b2=0.999, eps=1e-8, t2=0):
    """Construct optimizer triple for Adam.

    Args:
      step_size: positive scalar, or a callable representing a step size schedule
        that maps the iteration index to positive scalar.
      b1: optional, a positive scalar value for beta_1, the exponential decay rate
        for the first moment estimates (default 0.9).
      b2: optional, a positive scalar value for beta_2, the exponential decay rate
        for the second moment estimates (default 0.999).
      eps: optional, a positive scalar value for epsilon, a small constant for
        numerical stability (default 1e-8).

    Returns:
      An (init_fun, update_fun, get_params) triple.
    """
    step_size = make_schedule(step_size)

    def init(x0):
        m0 = jnp.zeros_like(x0)
        v0 = jnp.zeros_like(x0)
        return x0, m0, v0

    def update(i, g, state):
        x, m, v = state
        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * jnp.square(g) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1 ** (i + 1))  # Bias correction.
        vhat = v / (1 - b2 ** (i + 1))
        x = x - step_size(i) * mhat / (jnp.sqrt(vhat) + eps)
        # if i > t2:
        #     x = (jnp.abs(x) > 0.001) * x
        return x, m, v

    def get_params(state):
        x, _, _ = state
        return x
    return init, update, get_params
