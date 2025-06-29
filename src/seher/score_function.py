"""Implementation of score function gradient estimation.

The score function gradient estimator (also known as REINFORCE) estimates
gradients for expectations of the form E[f(x)] where x ~ p(x; θ) using:

∇_θ E[f(x)] = E[f(x) * ∇_θ log p(x; θ)]

This is useful when f(x) is not differentiable but p(x; θ) is, allowing
gradient estimation through sampling.
"""

import functools
from typing import Callable

import jax
import jax.lax as jl
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt

from .types import JaxRandomKey, ObjectiveFunction


def score_function_search_direction[
    ParameterDistribution,
    Parameter,
    ProblemData,
    Auxiliary,
](
    sample_fn: Callable[[ParameterDistribution, JaxRandomKey], Parameter],
    log_prob_fn: Callable[[Parameter, ParameterDistribution], jax.Array],
    objective: ObjectiveFunction[Parameter, ProblemData, Auxiliary],
    parameter_dist: ParameterDistribution,
    problem_data: ProblemData,
    key: JaxRandomKey,
    n_samples: int = 1,
    baseline_fn: Callable[[jax.Array], jax.Array] | None = None,
) -> tuple[ParameterDistribution, Auxiliary | None]:
    """Return a score function gradient estimate to decrease expected loss.

    Computes ∇_θ E[f(x)] ≈ f(x) * ∇_θ log p(x; θ) where:
    - f(x) is the loss function (possibly non-differentiable)
    - p(x; θ) is a parameterized distribution
    - x ~ p(x; θ) is a sample from the distribution

    Parameters
    ----------
    sample_fn:
        Function that takes parameters and key, returns a sample.
    log_prob_fn:
        Function that takes (sample, parameters) and returns
        log p(sample; parameters). Must be differentiable w.r.t. parameters.
    objective:
        ObjectiveFunction instance. This can be non-differentiable.
    parameter_dist:
        An abstract representation of the parameter distribution, which is what
        we will optimize. `sample_fn` has to turn it into a Parameter that
        `objective` understands, and `log_prob_fn` has to be able to give log
        probabilities for it.
    problem_data:
        Additional data passed to the objecive.
    key:
        RNG for sampling.
    n_samples:
        Number of samples to use for gradient estimation.
    baseline_fn:
        Optional function to compute baseline b(f) for variance reduction.
        Applied as: (f(x) - b(f)) * ∇_θ log p(x; θ).
        If None, no baseline is used.

    Returns
    -------
    gradient:
        Score function gradient estimate with the same shape as `parameter`.
    auxiliary:
        Auxiliary data from the loss evaluation of the first sample.

    """
    keys = jr.split(key, n_samples)

    @jax.vmap
    def sample_and_evaluate(k):
        parameter_sample = sample_fn(parameter_dist, k)
        loss_value, aux = objective(parameter_sample, problem_data, k)
        return parameter_sample, loss_value, aux

    parameter_samples, loss_values, auxs = sample_and_evaluate(keys)

    if baseline_fn is not None:
        baseline = baseline_fn(loss_values)
        weighted_losses = loss_values - baseline
    else:
        weighted_losses = loss_values

    @jax.vmap
    def compute_log_prob_grad(parameter_sample, weighted_loss):
        def log_prob_fn_for_grad(params):
            """Compute log prob of fixed parameter sample under params."""
            # Stop gradients to prevent flow through the random sample.
            fixed_parameter_sample = jl.stop_gradient(parameter_sample)
            return log_prob_fn(fixed_parameter_sample, params)

        log_prob_grad = jax.grad(log_prob_fn_for_grad)(parameter_dist)

        return jt.map(lambda grad: grad * weighted_loss, log_prob_grad)

    sample_gradients = compute_log_prob_grad(
        parameter_samples, weighted_losses
    )

    score_gradient = jt.map(lambda grads: grads.mean(axis=0), sample_gradients)

    # Return first auxiliary for consistency.
    # Handle case where auxs is a pytree from vmap
    first_aux = jax.tree.map(lambda x: x[0] if x is not None else None, auxs)
    return score_gradient, first_aux


def score_function_grad[
    ParameterDistribution,
    Parameter,
    ProblemData,
    Auxiliary,
](
    sample_fn: Callable[[ParameterDistribution, JaxRandomKey], Parameter],
    log_prob_fn: Callable[[Parameter, ParameterDistribution], jax.Array],
    objective: ObjectiveFunction[Parameter, ProblemData, Auxiliary],
    n_samples: int = 1,
    baseline_fn: Callable[[jax.Array], jax.Array] | None = None,
    has_aux: bool = False,
) -> Callable:
    """Return a gradient function based on score function gradient estimation.

    This is a convenience wrapper around `score_function_search_direction` such
    that it can serve as a substitute for `jax.grad`.

    Parameters
    ----------
    sample_fn:
        Function that samples from the distribution.
    log_prob_fn:
        Function that computes log probability of a sample.
    objective:
        ObjectiveFunction instance. This can be non-differentiable.
    n_samples:
        Number of samples to use for gradient estimation.
    baseline_fn:
        Optional baseline function for variance reduction.
    has_aux:
        Whether to also return auxiliary data from the loss.

    Returns
    -------
    Callable
        Given `parameter`, `problem_data`, and `key`, returns a score function
        gradient estimate.

    """
    grad_with_aux = functools.partial(
        score_function_search_direction,
        sample_fn,
        log_prob_fn,
        objective,
        n_samples=n_samples,
        baseline_fn=baseline_fn,
    )

    def grad_without_aux(*args, **kwargs):
        gradient, _ = grad_with_aux(*args, **kwargs)
        return gradient

    result = grad_with_aux if has_aux else grad_without_aux
    return result


def score_function_value_and_grad[
    ParameterDistribution,
    Parameter,
    ProblemData,
    Auxiliary,
](
    objective: ObjectiveFunction[Parameter, ProblemData, Auxiliary],
    sample_fn: Callable[[ParameterDistribution, JaxRandomKey], Parameter],
    log_prob_fn: Callable[[Parameter, ParameterDistribution], jax.Array],
    n_samples: int = 1,
    baseline_fn: Callable | None = None,
    has_aux: bool = True,
):
    """Return value and gradient function based on score function estimation.

    This is a convenience wrapper such that it can serve as a substitute for
    `jax.value_and_grad`.

    Parameters
    ----------
    objective:
        Objective function to minimize.
    sample_fn:
        Function that samples from the distribution.
    log_prob_fn:
        Function that computes log probability of a sample.
    n_samples:
        Number of samples to use for gradient estimation.
    baseline_fn:
        Optional baseline function for variance reduction.
    has_aux:
        If `True`, assume that `objective` returns a pair of which the first
        is the value and the second is auxiliary data.

    Returns
    -------
    Callable
        Function compatible with jax.value_and_grad signature.

    """
    grad_func = score_function_grad(
        sample_fn,
        log_prob_fn,
        objective,
        n_samples,
        baseline_fn,
        has_aux=False,
    )

    def inner(parameter_dist, problem_data, key):
        grad_key, value_key = jr.split(key, 2)

        grad = grad_func(
            parameter_dist=parameter_dist,
            problem_data=problem_data,
            key=grad_key,
        )

        parameter_sample = sample_fn(parameter_dist, value_key)
        value, aux = objective(parameter_sample, problem_data, value_key)

        if has_aux:
            return (value, aux), grad
        else:
            return value, grad

    return inner


def mean_baseline(values: jax.Array) -> jax.Array:
    """Return simple baseline that is the mean of values."""
    return values.mean()


def zero_baseline(values: jax.Array) -> jax.Array:
    """Return zero as a  baseline."""
    return jnp.zeros_like(values)
