"""Tests for policy search algorithms."""

import functools

import jax.random as jr
import pytest

from seher.ars import ars_grad, ars_value_and_grad
from seher.control.policy_search import (
    MCPSCLIProgressCallback,
)
from seher.control.solvers import PolicySearchSolver, plot_simulation_costs
from seher.systems.mujoco_playground import MujocoPlaygroundMDP
from seher.systems.pendulum import Pendulum


@pytest.mark.parametrize(
    "solver,problem,target,key",
    [
        pytest.param(
            PolicySearchSolver(
                obs_to_array=lambda state: state.cos_sin_repr(),
                n_simulations=32,
                steps_per_update=64,
                episode_length=64,
                optax_optimizer_kws={"learning_rate": 3e-3},
                max_updates=15_000,
                updates_per_eval=1_000,
            ),
            Pendulum(),
            0.5,
            jr.PRNGKey(13),
            marks=pytest.mark.performance,
        ),
        pytest.param(
            PolicySearchSolver(
                obs_to_array=lambda state: state.cos_sin_repr(),
                n_simulations=32,
                steps_per_update=64,
                episode_length=64,
                optax_optimizer_kws={"learning_rate": 3e-3},
                optimizer_kws={"value_and_grad": None},
                max_updates=15_000,
                updates_per_eval=1_000,
            ),
            Pendulum(),
            0.5,
            jr.PRNGKey(13),
            marks=[pytest.mark.performance],
        ),
        pytest.param(
            PolicySearchSolver(
                obs_to_array=lambda state: state.cos_sin_repr(),
                n_simulations=32,
                steps_per_update=16,
                episode_length=128,
                optax_optimizer_kws={"learning_rate": 3e-3},
                max_updates=20_000,
                updates_per_eval=1_000,
            ),
            Pendulum(),
            0.5,
            jr.PRNGKey(12),
            marks=[pytest.mark.slow, pytest.mark.performance],
        ),
        pytest.param(
            PolicySearchSolver(
                obs_to_array=lambda state: state.cos_sin_repr(),
                n_simulations=32,
                steps_per_update=16,
                episode_length=128,
                optax_optimizer_kws={"learning_rate": 3e-3},
                optimizer_kws={
                    "grad": functools.partial(
                        ars_grad, std=0.2, n_perturbations=6, top_k=2
                    )
                },
                max_updates=35_000,
                updates_per_eval=1_000,
            ),
            Pendulum(),
            0.5,
            jr.PRNGKey(12),
            marks=[pytest.mark.slow, pytest.mark.performance],
        ),
        pytest.param(
            PolicySearchSolver(
                obs_to_array=lambda state: state.cos_sin_repr(),
                n_simulations=32,
                steps_per_update=64,
                episode_length=64,
                optax_optimizer_kws={"learning_rate": 3e-3},
                optimizer_kws={
                    "value_and_grad": functools.partial(
                        ars_value_and_grad, std=0.2, n_perturbations=6, top_k=2
                    )
                },
                max_updates=35_000,
                updates_per_eval=1_000,
            ),
            Pendulum(),
            0.5,
            jr.PRNGKey(13),
            marks=pytest.mark.performance,
        ),
        pytest.param(
            PolicySearchSolver(
                obs_to_array=lambda state: state.cos_sin_repr(),
                n_simulations=32,
                steps_per_update=64,
                episode_length=64,
                optax_optimizer_kws={"learning_rate": 3e-3},
                optimizer_kws={
                    "grad": functools.partial(
                        ars_grad, std=0.2, n_perturbations=6, top_k=2
                    ),
                    "has_aux": True,
                },
                max_updates=25_000,
                updates_per_eval=1_000,
            ),
            Pendulum(),
            0.5,
            jr.PRNGKey(13),
            marks=pytest.mark.performance,
        ),
        pytest.param(
            PolicySearchSolver(
                obs_to_array=lambda state: state.obs,
                n_simulations=2,
                steps_per_update=32,
                episode_length=32,
                optax_optimizer_kws={"learning_rate": 3e-3},
                optimizer_kws={
                    "grad": functools.partial(
                        ars_grad, std=0.2, n_perturbations=6, top_k=2
                    ),
                    "has_aux": True,
                },
                max_updates=25_000,
                updates_per_eval=1_000,
            ),
            MujocoPlaygroundMDP.from_registry("CartpoleBalance").replace(  # type: ignore
                n_inner_steps=4
            ),
            0.5,
            jr.PRNGKey(12),
            marks=[pytest.mark.slow, pytest.mark.performance],
        ),
    ],
    ids=[
        "pendulum",
        "pendulum_grad",
        "pendulum_non_episodic",
        "pendulum_non_episodic_ars_grad",
        "pendulum_ars_value_and_grad",
        "pendulum_ars_grad",
        "cartpole_balance_ars_grad",
    ],
)
def test_policy_search_on_mdp(
    solver: PolicySearchSolver,
    problem,
    target: float,
    key,
):
    """Test if `solver` reaches `target` cost."""
    # Add progress callback like original tests
    progress_callback = MCPSCLIProgressCallback(total_steps=solver.max_updates)
    solver.callbacks.append(progress_callback)

    solver.solve(problem, key=key)

    # Evaluate with multiple simulations like original test
    history = solver.simulate(n_steps=100, n_simulations=8, key=key)

    # Plot the results like original test
    plot_simulation_costs(history)

    # Use same evaluation metric as original test: max cost in last 10 steps
    assert history.costs[:, -10:].max() < target
