"""Tests for optimizers."""

import functools

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

from seher.ars import ars_value_and_grad
from seher.score_function import mean_baseline, score_function_value_and_grad
from seher.stepper.mppi import GaussianMPPIOptimizer
from seher.stepper.optax import OptaxOptimizer
from seher.types import JaxRandomKey


def _quadratic_cost(
    parameter: jax.Array, problem_data: None, key: JaxRandomKey
):
    del problem_data, key
    return (parameter**2).sum(), None


def _gaussian_sample(mean: jax.Array, key: JaxRandomKey):
    """Sample from N(mean, I)."""
    return mean + jr.normal(key, mean.shape)


def _gaussian_log_prob(sample: jax.Array, mean: jax.Array):
    """Compute log probability of sample under N(mean, I)."""
    return -0.5 * ((sample - mean) ** 2).sum()


def _quadratic_loss_on_sample(sample: jax.Array, problem_data: None):
    """Quadratic loss function that operates on samples."""
    del problem_data
    return (sample**2).sum(), None


@pytest.mark.parametrize(
    "optimizer,n_iter,target_cost",
    [
        (
            OptaxOptimizer(
                optimizer=optax.adam(3e-2), objective=_quadratic_cost
            ),
            100,
            0.01,
        ),
        (
            OptaxOptimizer(
                optimizer=optax.adam(3e-2),
                objective=_quadratic_cost,
                value_and_grad=functools.partial(
                    ars_value_and_grad,
                    std=0.5,
                    n_perturbations=128,
                    top_k=64,
                ),
                has_aux=True,
            ),
            100,
            0.01,
        ),
        (
            GaussianMPPIOptimizer(
                n_candidates=64,
                top_k=4,
                initial_loc=jnp.array(0.0),
                initial_scale=jnp.array(1.0),
                objective=_quadratic_cost,
                temperature=0.0,
            ),
            100,
            0.01,
        ),
        (
            OptaxOptimizer(
                optimizer=optax.adam(3e-2),
                objective=_quadratic_cost,
                value_and_grad=functools.partial(
                    score_function_value_and_grad,
                    sample_fn=_gaussian_sample,
                    log_prob_fn=_gaussian_log_prob,
                    n_samples=32,
                    baseline_fn=None,
                ),
                has_aux=True,
            ),
            200,
            0.1,
        ),
        (
            OptaxOptimizer(
                optimizer=optax.adam(3e-2),
                objective=_quadratic_cost,
                value_and_grad=functools.partial(
                    score_function_value_and_grad,
                    sample_fn=_gaussian_sample,
                    log_prob_fn=_gaussian_log_prob,
                    n_samples=32,
                    baseline_fn=mean_baseline,
                ),
                has_aux=True,
            ),
            200,
            0.1,
        ),
    ],
)
def test_optimizer_on_quadratic(optimizer, n_iter, target_cost):
    """Test optimizer against a target cost."""
    carry = optimizer.initial_carry(jnp.ones(2))
    key = jr.PRNGKey(0)
    for _ in range(n_iter):
        key, subkey = jr.split(key)
        carry, params, _ = optimizer(carry, None, subkey)

    cost, _ = optimizer.objective(params, None, None)
    assert cost < target_cost
