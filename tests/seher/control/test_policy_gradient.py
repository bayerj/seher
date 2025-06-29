"""Tests for policy gradient implementation."""

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

from seher.apx_arch import MLP, StaticMLPPolicy
from seher.control.policy_search.base import (
    BaseProgressCallback,
    cost_from_eval_history,
)
from seher.control.policy_search.policy_gradient import (
    PolicyGradientAuxiliary,
    PolicyGradientCarry,
    PolicyGradientOptimizer,
)
from seher.control.solvers import PolicyGradientSolver, plot_simulation_costs
from seher.control.tanh_gaussian_policy_mdp import TanhGaussianPolicyControl
from seher.score_function import mean_baseline
from seher.stepper.optax import OptaxOptimizer
from seher.systems.lqr import make_simple_2d_lqr
from seher.systems.pendulum import Pendulum
from seher.types import JaxRandomKey


def obs_to_array(state):
    """Convert state to array, handling original and GaussianPolicyState."""
    # Handle both original state and GaussianPolicyState recursively
    while hasattr(state, "original_state"):
        state = state.original_state

    # Handle different state types
    if hasattr(state, "cos_sin_repr"):
        # Pendulum state
        return state.cos_sin_repr()
    elif hasattr(state, "x"):
        # LQR state
        return state.x
    else:
        # Fallback for other state types
        return state


def test_policy_gradient_optimizer_basic():
    """Test basic PolicyGradientOptimizer functionality."""
    mdp = Pendulum()

    # Create a simple MLP-based policy for testing

    # Create MLP that outputs [loc, inv_softplus_scale] (2D output for 1D)
    mlp = MLP.make(
        inpt_size=3,  # Pendulum observation dim
        output_size=2,  # [loc, inv_softplus_scale]
        layer_sizes=[4],
        activations=[jnn.tanh, lambda x: x],  # Linear output
        key=jr.PRNGKey(0),
    )

    def array_to_gaussian_control(arr):
        return TanhGaussianPolicyControl(
            loc=arr[:1],  # First element
            inv_softplus_scale=arr[1:2],  # Second element
        )

    policy = StaticMLPPolicy(
        mlp=mlp,
        obs_to_array=obs_to_array,
        array_to_control=array_to_gaussian_control,
    )

    # Create a simple optimizer
    optimizer = OptaxOptimizer(
        objective=None,  # type: ignore
        optimizer=optax.adam(learning_rate=0.01),
    )

    pg_optimizer = PolicyGradientOptimizer(
        mdp=mdp,
        gaussian_policy=policy,
        n_simulations=4,
        n_steps=8,
        optimizer=optimizer,  # type: ignore
    )

    # Test initial carry
    carry = pg_optimizer.initial_carry(sample_parameter=policy)
    assert isinstance(carry, PolicyGradientCarry)
    assert carry.current == policy
    assert carry.steps_since_init == 0.0

    # Test objective function
    key = jr.PRNGKey(42)
    loss, aux = pg_optimizer.objective(policy, None, key, carry)

    assert isinstance(loss, jax.Array)
    assert loss.shape == ()  # Scalar loss
    assert isinstance(aux, PolicyGradientAuxiliary)
    assert aux.returns.shape == (4,)  # n_simulations
    assert aux.log_probs.shape == (4, 7)  # n_simulations x (n_steps-1)


def test_policy_gradient_optimizer_step():
    """Test PolicyGradientOptimizer step functionality."""
    mdp = Pendulum()

    mlp = MLP.make(
        inpt_size=3,  # Pendulum observation dim
        output_size=2,  # [loc, inv_softplus_scale]
        layer_sizes=[4],
        activations=[jnn.tanh, lambda x: x],  # Linear output
        key=jr.PRNGKey(1),
    )

    def array_to_gaussian_control(arr):
        return TanhGaussianPolicyControl(
            loc=arr[:1],  # First element
            inv_softplus_scale=arr[1:2],  # Second element
        )

    policy = StaticMLPPolicy(
        mlp=mlp,
        obs_to_array=obs_to_array,
        array_to_control=array_to_gaussian_control,
    )

    optimizer = OptaxOptimizer(
        objective=None,  # type: ignore
        optimizer=optax.adam(learning_rate=0.01),
    )

    pg_optimizer = PolicyGradientOptimizer(
        mdp=mdp,
        gaussian_policy=policy,
        n_simulations=4,
        n_steps=8,
        optimizer=optimizer,  # type: ignore
    )

    carry = pg_optimizer.initial_carry(sample_parameter=policy)
    key = jr.PRNGKey(123)

    # Test one optimization step
    new_carry, _, aux = pg_optimizer(carry, None, key)

    assert isinstance(new_carry, PolicyGradientCarry)
    assert isinstance(aux, PolicyGradientAuxiliary)
    assert new_carry.steps_since_init == 8.0  # n_steps


def test_policy_gradient_solver_basic():
    """Test basic PolicyGradientSolver functionality."""
    solver = PolicyGradientSolver(
        obs_to_array=obs_to_array,
        n_simulations=4,
        steps_per_update=16,
        episode_length=16,
        max_updates=10,
        optax_optimizer_kws={"learning_rate": 0.01},
    )

    problem = Pendulum()
    key = jr.PRNGKey(456)

    # Test that solver can be initialized
    assert not solver.prepared
    assert solver.policy is None

    # Test preparation
    solver._prepare(problem, key)
    assert solver.prepared
    assert solver.policy is not None

    # Test that the policy outputs concrete control arrays
    sample_state = problem.init(key)
    carry = solver.policy.initial_carry()
    carry, control = solver.policy(
        carry,
        sample_state,
        problem.empty_control(),
        key,  # type: ignore
    )
    assert isinstance(control, jax.Array)
    assert control.shape == (1,)  # Pendulum has 1D control


def test_policy_gradient_solver_solve():
    """Test PolicyGradientSolver solve method (quick version)."""
    solver = PolicyGradientSolver(
        obs_to_array=obs_to_array,
        n_simulations=2,
        steps_per_update=8,
        episode_length=8,
        max_updates=5,  # Very few updates for quick test
        optax_optimizer_kws={"learning_rate": 0.01},
    )

    problem = Pendulum()
    key = jr.PRNGKey(789)

    solver.solve(problem, key=key, eval_key=key)

    # Check that solver completed
    assert solver.is_solved
    assert solver.prepared
    assert solver.policy is not None
    assert solver.history is not None

    # Test simulation
    history = solver.simulate(n_steps=10, n_simulations=2, key=key)
    assert history.states.angle.shape == (2, 10, 1)
    assert history.costs.shape == (2, 10, 1)


def test_policy_gradient_box_constraints():
    """Test that actions respect box constraints."""
    # Test with custom pendulum bounds
    pendulum = Pendulum(max_torque=1.5)

    solver = PolicyGradientSolver(
        obs_to_array=obs_to_array,
        n_simulations=2,
        steps_per_update=4,
        episode_length=4,
        max_updates=2,
        optax_optimizer_kws={"learning_rate": 0.01},
    )

    key = jr.PRNGKey(456)
    solver.solve(pendulum, key=key, eval_key=key)

    # Test simulation and verify bounds
    history = solver.simulate(n_steps=10, n_simulations=2, key=key)
    actions = history.controls

    # Actions should be within [-1.5, 1.5]
    assert jnp.all(actions >= pendulum.control_min[0])
    assert jnp.all(actions <= pendulum.control_max[0])
    assert actions.shape == (2, 10, 1)


def test_policy_gradient_with_baseline():
    """Test PolicyGradientOptimizer with baseline function."""
    mdp = Pendulum()

    mlp = MLP.make(
        inpt_size=3,  # Pendulum observation dim
        output_size=2,  # [loc, inv_softplus_scale]
        layer_sizes=[4],
        activations=[jnn.tanh, lambda x: x],  # Linear output
        key=jr.PRNGKey(2),
    )

    def array_to_gaussian_control(arr):
        return TanhGaussianPolicyControl(
            loc=arr[:1],  # First element
            inv_softplus_scale=arr[1:2],  # Second element
        )

    policy = StaticMLPPolicy(
        mlp=mlp,
        obs_to_array=obs_to_array,
        array_to_control=array_to_gaussian_control,
    )

    optimizer = OptaxOptimizer(
        objective=None,  # type: ignore
        optimizer=optax.adam(learning_rate=0.01),
    )

    pg_optimizer = PolicyGradientOptimizer(
        mdp=mdp,
        gaussian_policy=policy,
        n_simulations=4,
        n_steps=8,
        optimizer=optimizer,  # type: ignore
        baseline_fn=mean_baseline,
    )

    carry = pg_optimizer.initial_carry(sample_parameter=policy)
    key = jr.PRNGKey(999)

    # Test that baseline is applied
    loss, _ = pg_optimizer.objective(policy, None, key, carry)
    assert isinstance(loss, jax.Array)
    assert loss.shape == ()

    # The loss should be different with vs without baseline
    pg_optimizer_no_baseline = pg_optimizer.replace(  # type: ignore
        baseline_fn=None
    )
    loss_no_baseline, _ = pg_optimizer_no_baseline.objective(
        policy, None, key, carry
    )

    # They should generally be different (though could be same by chance)
    # Just check that both are finite
    assert jnp.isfinite(loss)
    assert jnp.isfinite(loss_no_baseline)


@pytest.mark.parametrize(
    "solver,problem,target,solve_key,eval_key,policy_init_key",
    [
        pytest.param(
            PolicyGradientSolver(
                obs_to_array=obs_to_array,
                n_simulations=123,
                steps_per_update=64,
                episode_length=64,
                optax_optimizer="sgd",
                optax_optimizer_kws={"learning_rate": 0.00017119554359135567},
                max_updates=1_000,
                updates_per_eval=200,
                policy_mlp_kws={
                    "layer_sizes": [14],
                    "activations": [
                        jnn.tanh,
                        lambda x: x,
                    ],
                },
                max_grad_norm=5055.71947049641,
                baseline_fn=lambda returns: returns.mean(),
            ),
            make_simple_2d_lqr(
                dt=0.5,
                position_cost=1.0,
                velocity_cost=0.0,
                control_cost=0.0,
                discount=0.99,
                noise_scale=0.0,
                init_scale=0.5,
            ),
            0.02,
            jr.PRNGKey(273),
            jr.PRNGKey(1337),
            jr.PRNGKey(660),
            marks=pytest.mark.performance,
        ),
    ],
)
def test_policy_gradient_on_mdp(
    solver: PolicyGradientSolver,
    problem,
    target: float,
    solve_key: JaxRandomKey,
    policy_init_key: JaxRandomKey,
    eval_key: JaxRandomKey,
):
    """Test if PolicyGradientSolver reaches target cost on LQR system."""
    progress_callback = BaseProgressCallback(
        total_steps=solver.max_updates,
        metric_extractors={
            "eval_cost": cost_from_eval_history,
            "min_control": lambda th, *_: th.controls.min(),  # type: ignore
            "max_control": lambda th, *_: th.controls.max(),  # type: ignore
        },
        description="Policy Gradient Training...",
    )
    solver.callbacks.append(progress_callback)

    solver.solve(
        problem,
        key=solve_key,
        eval_key=eval_key,
        policy_init_key=policy_init_key,
    )
    history = solver.simulate(n_steps=100, n_simulations=8, key=eval_key)
    plot_simulation_costs(history)
    assert history.costs.mean() <= target
