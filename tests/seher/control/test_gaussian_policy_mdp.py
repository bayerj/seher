"""Tests for TanhGaussianPolicyMDP."""

import jax
import jax.numpy as jnp
import jax.random as jr

from seher.control.tanh_gaussian_policy_mdp import (
    TanhGaussianPolicyControl,
    TanhGaussianPolicyMDP,
    TanhGaussianPolicyState,
    compute_tanh_gaussian_policy_log_probs,
)
from seher.simulate import simulate
from seher.systems.lqr import make_simple_2d_lqr
from seher.systems.pendulum import Pendulum


def test_tanh_gaussian_policy_control():
    """Test TanhGaussianPolicyControl basic functionality."""
    control = TanhGaussianPolicyControl(
        loc=jnp.array([1.0, 2.0]),
        inv_softplus_scale=jnp.array([0.0, 1.0]),
    )

    assert control.loc.shape == (2,)
    assert control.scale.shape == (2,)
    assert jnp.allclose(control.scale[0], jax.nn.softplus(-1.0), atol=1e-4)
    assert jnp.allclose(control.scale[1], jax.nn.softplus(0.0), atol=1e-4)


def test_tanh_gaussian_policy_mdp_basic():
    """Test basic TanhGaussianPolicyMDP functionality."""
    original_mdp = Pendulum()
    gaussian_mdp = TanhGaussianPolicyMDP(original_mdp=original_mdp)

    assert gaussian_mdp.discount == original_mdp.discount

    empty_control = gaussian_mdp.empty_control()
    assert isinstance(empty_control, TanhGaussianPolicyControl)
    assert empty_control.loc.shape == original_mdp.empty_control().shape
    assert (
        empty_control.inv_softplus_scale.shape
        == original_mdp.empty_control().shape
    )

    key = jr.PRNGKey(0)
    initial_state = gaussian_mdp.init(key)
    assert isinstance(initial_state, TanhGaussianPolicyState)
    assert (
        initial_state.last_control.shape == original_mdp.empty_control().shape
    )


def test_tanh_gaussian_policy_mdp_transition():
    """Test TanhGaussianPolicyMDP transition functionality."""
    original_mdp = Pendulum()
    gaussian_mdp = TanhGaussianPolicyMDP(original_mdp=original_mdp)

    key = jr.PRNGKey(42)
    init_key, transit_key = jr.split(key)

    state = gaussian_mdp.init(init_key)

    control = TanhGaussianPolicyControl(
        loc=jnp.array([0.5]),
        inv_softplus_scale=jnp.array([0.0]),
    )

    new_state = gaussian_mdp.transit(state, control, transit_key)
    assert isinstance(new_state, TanhGaussianPolicyState)
    assert new_state.last_control.shape == (1,)

    cost = gaussian_mdp.cost(new_state, control, transit_key)
    assert cost.shape == (1,)


def test_tanh_gaussian_policy_mdp_simulation():
    """Test running simulation with TanhGaussianPolicyMDP."""
    original_mdp = Pendulum()
    gaussian_mdp = TanhGaussianPolicyMDP(original_mdp=original_mdp)

    class ConstantTanhGaussianPolicy:
        def initial_carry(self):
            return None

        def __call__(self, carry, obs, control, key):
            del carry, obs, control, key
            constant_control = TanhGaussianPolicyControl(
                loc=jnp.array([0.1]),
                inv_softplus_scale=jnp.array([-1.0]),
            )
            return None, constant_control

    policy = ConstantTanhGaussianPolicy()
    key = jr.PRNGKey(123)

    history = simulate(gaussian_mdp, policy, n_steps=5, key=key)

    assert history.states.original_state.angle.shape == (5, 1)
    assert history.states.last_control.shape == (5, 1)
    assert history.controls.loc.shape == (5, 1)
    assert history.controls.inv_softplus_scale.shape == (5, 1)
    assert history.costs.shape == (5, 1)


def test_compute_tanh_gaussian_policy_log_probs():
    """Test computation of log probabilities from history."""
    original_mdp = Pendulum()
    gaussian_mdp = TanhGaussianPolicyMDP(original_mdp=original_mdp)

    class ConstantTanhGaussianPolicy:
        def initial_carry(self):
            return None

        def __call__(self, carry, obs, control, key):
            del carry, obs, control, key
            constant_control = TanhGaussianPolicyControl(
                loc=jnp.array([0.0]),
                inv_softplus_scale=jnp.array([0.0]),
            )
            return None, constant_control

    policy = ConstantTanhGaussianPolicy()
    key = jr.PRNGKey(456)

    keys = jr.split(key, 3)
    histories = []
    for k in keys:
        history = simulate(gaussian_mdp, policy, n_steps=4, key=k)
        histories.append(history)

    batch_history = jax.tree.map(lambda *xs: jnp.stack(xs), *histories)

    log_probs = compute_tanh_gaussian_policy_log_probs(
        batch_history, original_mdp.control_min, original_mdp.control_max
    )

    assert log_probs.shape == (3, 3)
    assert jnp.all(jnp.isfinite(log_probs))


def test_mdp_equivalence_timing_issue():
    """Regression test for a timing issue in TanhGaussianPolicyMDP.

    This test compares using the original MDP with a deterministic policy
    vs the TanhGaussianPolicyMDP. They should produce equivalent costs,
    but the current implementation uses the wrong control timing.
    """
    # Create deterministic LQR system for clear comparison
    original_mdp = make_simple_2d_lqr(
        dt=0.1,
        position_cost=1.0,
        velocity_cost=0.0,
        control_cost=0.1,
        max_control=1.0,
        noise_scale=0.0,  # No noise for deterministic comparison
        init_scale=0.0,  # Start at origin
    )

    gaussian_mdp = TanhGaussianPolicyMDP(original_mdp=original_mdp)

    # Create deterministic policies that use the same control sequence
    class FixedPolicy:
        def __init__(self, controls):
            self.controls = jnp.array(controls)

        def initial_carry(self):
            return 0  # Step counter

        def __call__(self, carry, obs, control, key):
            del obs, key
            step = carry
            # Use JAX-compatible indexing with clipping
            safe_step = jnp.clip(step, 0, len(self.controls) - 1)
            action = jnp.where(
                step < len(self.controls),
                self.controls[safe_step],
                jnp.zeros_like(control),
            )
            return step + 1, action

    class FixedTanhGaussianPolicy:
        def __init__(self, controls, control_min, control_max):
            self.controls = jnp.array(controls)
            self.control_min = control_min
            self.control_max = control_max

        def initial_carry(self):
            return 0

        def __call__(self, carry, obs, control, key):
            del obs, control, key
            step = carry
            safe_step = jnp.clip(step, 0, len(self.controls) - 1)

            # Get target control
            target_control = jnp.where(
                step < len(self.controls),
                self.controls[safe_step],
                jnp.zeros_like(
                    self.controls[0]
                ),  # Use first control as template
            )

            # Convert from control space to pre-tanh space
            normalized = (
                2
                * (target_control - self.control_min)
                / (self.control_max - self.control_min)
                - 1
            )
            pre_tanh = jnp.arctanh(jnp.clip(normalized, -0.99, 0.99))

            action = TanhGaussianPolicyControl(
                loc=pre_tanh,
                inv_softplus_scale=jnp.full_like(
                    pre_tanh, -10.0
                ),  # Very small scale
            )
            return step + 1, action

    # Fixed control sequence
    fixed_controls = [
        jnp.array([0.5]),
        jnp.array([-0.3]),
        jnp.array([0.1]),
        jnp.array([0.0]),
    ]

    fixed_policy = FixedPolicy(fixed_controls)
    fixed_gaussian_policy = FixedTanhGaussianPolicy(
        fixed_controls, original_mdp.control_min, original_mdp.control_max
    )

    key = jr.PRNGKey(42)

    # Simulate both
    original_history = simulate(original_mdp, fixed_policy, n_steps=5, key=key)
    gaussian_history = simulate(
        gaussian_mdp, fixed_gaussian_policy, n_steps=5, key=key
    )

    # Check that states evolve the same way
    original_states = original_history.states.x
    gaussian_states = gaussian_history.states.original_state.x

    # States should be very close (accounting for stochastic sampling)
    assert jnp.allclose(original_states, gaussian_states, atol=1e-4), (
        "States should evolve very similarly in both MDPs"
    )

    # Extract the actual controls that were applied
    original_controls = original_history.controls[
        :-1
    ]  # Last control not applied
    gaussian_controls = gaussian_history.states.last_control[
        1:
    ]  # Skip initial zero control

    print("Original controls applied:", original_controls.flatten())
    print("Gaussian controls applied:", gaussian_controls.flatten())
    print(
        "Gaussian last_control at each step:",
        gaussian_history.states.last_control.flatten(),
    )

    # Note: Controls may not match exactly due to the timing issue - this is
    # expected.
    print(
        "Control differences (expected due to timing offset):",
        (original_controls - gaussian_controls).flatten(),
    )

    original_costs = original_history.costs
    gaussian_costs = gaussian_history.costs

    assert jnp.allclose(original_costs, gaussian_costs, atol=1e-4)
