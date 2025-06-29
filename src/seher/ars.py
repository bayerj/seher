"""Implementation of Augmented Random Search.

ARS is an optimisation method that is based on finite differences. It finds
an approximate gradient. Described in [1]_.

.. [1] Mania, Horia, Aurelia Guy, and Benjamin Recht. "Simple random search
   provides a competitive approach to reinforcement learning." arXiv preprint
   arXiv:1803.07055 (2018).
"""

import functools
import operator
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt

from .types import JaxRandomKey, ObjectiveFunction


def ars_search_direction[Parameters, ProblemData, Auxiliary](
    loss: ObjectiveFunction[Parameters, ProblemData, Auxiliary],
    parameter: Parameters,
    problem_data: ProblemData,
    key: JaxRandomKey,
    n_perturbations: int,
    top_k: int,
    std: float,
) -> tuple[Parameters, Auxiliary]:
    """Return a search direction to decrease `loss`.

    Parameters
    ----------
    loss:        The loss function. Should accept a pytree of the same shape as
        `params`.
    parameter:
        A jax pytree which represents the space to search.
    n_perturbations:
        Amount of perturbation directions to pick randomly.
    top_k:
        Number of directions to mix, the best `top_k` are used.
    std:
        Standard deviation for the search directions.
    key:
        RNG for all downstream stochasticity.
    problem_data:
        Passed to `loss`.

    Returns
    -------
    Has the same shape as `params`.

    """
    treedef = jax.tree_util.tree_structure(parameter)
    num_leaves = treedef.num_leaves
    flat_pytree, treedef = jax.tree_util.tree_flatten(parameter)

    @jax.vmap
    def get_random_search_directions(key):
        keys = jr.split(key, num_leaves)

        def noise_as(leaf, key):
            noise = jr.normal(key=key, shape=leaf.shape) * std
            return noise

        flat_noisy = [noise_as(x, k) for k, x in zip(keys, flat_pytree)]
        search_direction = jax.tree_util.tree_unflatten(treedef, flat_noisy)

        return search_direction

    key, search_key = jax.random.split(key)
    search_directions = get_random_search_directions(
        jax.random.split(search_key, n_perturbations)
    )

    if False:

        @functools.partial(jax.vmap, in_axes=(0, None))
        def eval_search_directions(search_direction, key):
            left = jt.map(operator.add, parameter, search_direction)
            right = jt.map(operator.sub, parameter, search_direction)
            left_loss, _ = loss(
                parameter=left, problem_data=problem_data, key=key
            )
            right_loss, _ = loss(
                parameter=right, problem_data=problem_data, key=key
            )
            return left_loss, right_loss

        key, apply_key = jax.random.split(key)
        left_losses, right_losses = eval_search_directions(
            search_directions, apply_key
        )

    else:

        @functools.partial(jax.vmap, in_axes=(0, None))
        def eval_parameters(parameter, key):
            losses, auxs = loss(
                parameter=parameter, problem_data=problem_data, key=key
            )
            return losses, auxs

        left_candidates = jt.map(operator.add, parameter, search_directions)
        right_candidates = jt.map(operator.sub, parameter, search_directions)
        candidates = jt.map(
            lambda a, b: jnp.concatenate([a, b]),
            left_candidates,
            right_candidates,
        )
        losses, auxs = eval_parameters(candidates, key)
        left_losses, right_losses = jnp.split(losses, 2, 0)

    best_losses = jnp.minimum(left_losses, right_losses)
    _, top_k_idxs = jax.lax.top_k(-best_losses, k=top_k)

    # XXX While we average the search directions, it is unclear how to pick one
    # of the auxiliaries. Here we opt for picking the one of the best search,
    # but it might not bet what we always want.
    best_loss_idx = jnp.argmin(losses)
    best_aux = jax.tree.map(lambda leaf: leaf[best_loss_idx], auxs)

    best_left_losses = left_losses[top_k_idxs]
    best_right_losses = right_losses[top_k_idxs]
    cost_std = jnp.std(jnp.concatenate([best_left_losses, best_right_losses]))
    cost_std = jnp.maximum(cost_std, 1e-6)
    right_minus_left = right_losses - left_losses

    search_direction = jt.map(
        lambda sd: (
            -jnp.einsum(
                "n...,n->n...", sd[top_k_idxs], right_minus_left[top_k_idxs]
            )
        ).mean(0)
        / cost_std,
        search_directions,
    )

    if False:
        # XXX decide whether to use this instead
        _, aux = loss(
            parameter=parameter,
            problem_data=problem_data,
            key=key,
        )

    return search_direction, best_aux


def ars_grad[Parameters, ProblemData, Auxiliary](
    loss: ObjectiveFunction[Parameters, ProblemData, Auxiliary],
    n_perturbations: int,
    top_k: int,
    std: float,
    has_aux: bool = False,
) -> Callable:
    """Return a gradient function based on augmented random search.

    Possible also return auxiliary computation.

    This is a convenience wrapper around `ars_search_direction` such that it
    can serve as a substitute for `jax.grad`.

    Parameters
    ----------
    loss:
        Objective to minimize.
    n_perturbations: int
        Number of perturbation directions to sample.
    top_k: int
        Number of perturbation directions to average. The best, i.e.  minimal,
        `top_k` are kept.
    std: float
        Standard deviation for the search directions.
    has_aux:
        Whether to also return auxiliary data from the loss. Currently not
        implemented.

    Returns
    -------
    Callable
        Given `parameter`, `key`, and `problem_data`, returns an approximate
        gradient of `loss`.

    """
    grad_with_aux = functools.partial(
        ars_search_direction,
        loss,
        n_perturbations=n_perturbations,
        top_k=top_k,
        std=std,
    )

    def grad_without_aux(*args, **kwargs):
        gradient, aux = grad_with_aux(*args, **kwargs)
        return gradient

    result = grad_with_aux if has_aux else grad_without_aux
    return result


def ars_value_and_grad(
    objective,
    n_perturbations: int,
    top_k: int,
    std: float,
    has_aux: bool = True,
):
    """Return the value and gradient function based on augmented random search.

    This is a convenience wrapper around `ars_search_direction` such that it
    can serve as a substitute for `jax.grad`.

    If `has_aux` is True, then the result is `((value, auxiliary), gradient)`.

    Parameters
    ----------
    objective:
        Objective to minimize.
    n_perturbations: int
        Number of perturbation directions to sample.
    top_k: int
        Number of perturbation directions to average. The best, i.e.  minimal,
        `top_k` are kept.
    std: float
        Standard deviation for the search directions.
    has_aux:
        If `True`, assume that `objective` returns a pair of which the first
        is the value and the second is auxiliary data.

    Returns
    -------
    Callable
        Given `parameter`, `key`, and `problem_data`, returns the value of the
        objective, an approximate gradient of `objective` and auxiliary data
        if `has_aux` is True.

    """
    grad_func = ars_grad(objective, n_perturbations, top_k, std)

    def inner(parameter, problem_data, key):
        grad_key, value_key = jr.split(key, 2)
        grad = grad_func(
            parameter=parameter, problem_data=problem_data, key=grad_key
        )
        value, aux = objective(
            parameter=parameter, problem_data=problem_data, key=value_key
        )
        if has_aux:
            return (value, aux), grad
        else:
            return value, grad

    return inner
