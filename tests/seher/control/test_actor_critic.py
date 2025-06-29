"""Tests for actor-critic methods."""

import jax.random as jr
import optax
import pytest

from seher.control.policy_search import ActorCriticCLIProgressCallback
from seher.control.solvers import (
    ActorCriticEnsembleSolver,
    ActorCriticSolver,
    plot_simulation_costs,
)
from seher.systems.pendulum import (
    SparsePendulum,
    SparsePendulumSwingup,
)


@pytest.mark.parametrize(
    "solver,problem,target,key,policy_init_key,critic_init_key",
    [
        pytest.param(
            ActorCriticSolver(
                obs_to_array=lambda state: state.cos_sin_repr(),
                state_to_array=lambda state: state.cos_sin_repr(),
                n_simulations=32,
                steps_per_update=32,
                episode_length=32,
                td_lambda=0.95,
                critic_weight=1.0,
                optax_optimizer_kws={"learning_rate": 3e-3},
                max_updates=25_000,
            ),
            SparsePendulum(discount=0.95),
            -1.0,
            jr.PRNGKey(15),
            jr.PRNGKey(32),
            jr.PRNGKey(15),
            marks=pytest.mark.performance,
        ),
        pytest.param(
            ActorCriticSolver(
                obs_to_array=lambda state: state.cos_sin_repr(),
                state_to_array=lambda state: state.cos_sin_repr(),
                n_simulations=29,
                steps_per_update=16,
                episode_length=32,
                td_lambda=0.976,
                critic_weight=4.15,
                optax_optimizer_kws={"learning_rate": 0.016},
                optimizer_kws={
                    "optimizer": optax.chain(
                        optax.clip_by_global_norm(0.1),
                        optax.adam(0.016),
                    )
                },
                polyak_step_size=0.0018,
                max_updates=25_000,
            ),
            SparsePendulum(discount=0.98),
            -0.88,
            jr.PRNGKey(3),
            jr.PRNGKey(32),
            jr.PRNGKey(15),
            marks=pytest.mark.performance,
        ),
        pytest.param(
            ActorCriticEnsembleSolver(
                obs_to_array=lambda state: state.cos_sin_repr(),
                state_to_array=lambda state: state.cos_sin_repr(),
                n_simulations=32,
                steps_per_update=32,
                episode_length=32,
                td_lambda=0.95,
                critic_weight=1.0,
                optimism_coeff=1.0,
                n_ensemble_members=4,
                critic_mlp_kws={"layer_sizes": [16]},
                optax_optimizer_kws={"learning_rate": 3e-3},
                max_updates=25_000,
            ),
            SparsePendulumSwingup(discount=0.95),
            0.0,
            jr.PRNGKey(18),
            jr.PRNGKey(33),
            jr.PRNGKey(7),
            marks=pytest.mark.performance,
        ),
    ],
)
def test_actor_critic_on_mdp(
    solver: ActorCriticSolver,
    problem,
    target: float,
    key,
    policy_init_key,
    critic_init_key,
):
    """Test if `solver` reaches `target` cost."""
    progress_callback = ActorCriticCLIProgressCallback(
        total_steps=solver.max_updates
    )
    solver.callbacks.append(progress_callback)

    solver.solve(
        problem,
        key=key,
        policy_init_key=policy_init_key,
        critic_init_key=critic_init_key,
    )

    history = solver.simulate(n_steps=100, n_simulations=8, key=key)
    plot_simulation_costs(history)
    assert history.costs[:, -10:].mean() <= target
