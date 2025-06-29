"""Helpers for calculating TD-lambda returns."""

import jax
import jax.lax as jl


def td_lambda_return(
    costs: jax.Array, cost_to_gos: jax.Array, discount: float, td_lambda: float
) -> jax.Array:
    """Return TD-lambda returns.

    This is a recursive implementation that follows equation 4 in [1]_.

    Parameters
    ----------
    costs:
        Has shape `(T,)`.
    cost_to_gos:
        Has shape `(T,)`.
    discount: float
        Discount factor of the returns.
    td_lambda: float
        Lambda value for mixing different horizons.

    Returns
    -------
    Returns for every step. Has shape `(T,)`.

    References
    ----------

    .. [1] : Mastering Atari with Discrete World Models, Hafner et al, 2020.


    Examples
    --------
    >>> import jax.numpy as jnp
    >>> costs = jnp.array([0.1, 0.3, 0.1])
    >>> cost_to_gos = jnp.array([9.1, 9.4, 9.2])
    >>> td_lambda_return(costs, cost_to_gos, discount=0.9, td_lambda=0.95)
    Array([7.253655 , 7.8879004, 8.38     ], dtype=float32)

    """
    if not costs.shape == cost_to_gos.shape:
        raise ValueError(
            f"Differing shapes: {costs.shape=} and {cost_to_gos.shape=}"
        )
    (n_steps,) = cost_to_gos.shape
    initial = cost_to_gos[-1]
    inputs = costs + (1 - td_lambda) * discount * cost_to_gos

    def f(carry, inpt):
        next_ = inpt + discount * td_lambda * carry
        return next_, next_

    _, returns = jl.scan(f, initial, inputs[::-1], n_steps)
    returns = returns[::-1]
    return returns
