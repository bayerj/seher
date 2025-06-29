"""Tests for LQR system."""

import functools

import flax.struct
import jax
import jax.numpy as jnp
import jax.random as jr

from seher.simulate import simulate
from seher.systems.lqr import LQRState, make_simple_2d_lqr


@flax.struct.dataclass
class SimplePolicy:
    """A simple policy that just returns 0 as a control always."""

    def initial_carry(self):  # noqa: D102
        return None

    def __call__(self, carry, obs, control, key):  # noqa: D102
        return None, jnp.zeros(1)


def test_lqr_basic():
    """Test basic LQR functionality."""
    lqr = make_simple_2d_lqr()

    assert lqr.state_dim == 2
    assert lqr.control_dim == 1
    assert lqr.control_min.shape == (1,)
    assert lqr.control_max.shape == (1,)

    key = jr.PRNGKey(0)
    state = lqr.init(key)
    assert isinstance(state, LQRState)
    assert state.x.shape == (2,)

    control = lqr.empty_control()
    assert control.shape == (1,)

    new_state = lqr.transit(state, control, key)
    assert isinstance(new_state, LQRState)
    assert new_state.x.shape == (2,)

    cost = lqr.cost(state, control, key)
    assert cost.shape == (1,)


def test_lqr_jit_hashing_issue():
    """Test that demonstrates the JIT hashing issue with LQR."""
    lqr = make_simple_2d_lqr()

    policy = SimplePolicy()
    key = jr.PRNGKey(42)

    # This should work without JIT
    this_simulate = functools.partial(simulate, lqr)
    history = this_simulate(policy, n_steps=5, key=key)
    assert history.states.x.shape == (5, 2)
    assert history.controls.shape == (5, 1)
    assert history.costs.shape == (5, 1)

    # This is where the hashing issue occurs
    try:
        jit_simulate = jax.jit(this_simulate, static_argnums=(1,))
        jit_history = jit_simulate(policy, 5, key)
        print("JIT compilation succeeded!")
        assert jit_history.states.x.shape == (5, 2)
    except (TypeError, ValueError) as e:
        print(f"JIT compilation failed with error: {e}")
        # This is expected to fail currently
        assert False


def test_lqr_batch_simulation():
    """Test batch simulation with LQR - this reproduces the solver issue."""
    lqr = make_simple_2d_lqr()

    policy = SimplePolicy()
    key = jr.PRNGKey(42)
    n_simulations = 4
    n_steps = 10

    this_simulate = functools.partial(simulate, lqr)

    # This should work without JIT
    keys = jr.split(key, n_simulations)
    batch_histories = []
    for k in keys:
        history = this_simulate(policy, n_steps=n_steps, key=k)
        batch_histories.append(history)

    batch_history = jax.tree.map(lambda *xs: jnp.stack(xs), *batch_histories)
    assert batch_history.states.x.shape == (n_simulations, n_steps, 2)
    assert batch_history.controls.shape == (n_simulations, n_steps, 1)

    # This is the problematic JIT compilation from the solver
    try:
        jit_batch_simulate = jax.jit(
            jax.vmap(this_simulate, in_axes=(None, None, 0)),
            static_argnums=(1, 6, 7, 8, 9),
        )
        jit_batch_history = jit_batch_simulate(policy, n_steps, keys)
        print("Batch JIT compilation succeeded!")
        assert jit_batch_history.states.x.shape == (n_simulations, n_steps, 2)
    except (TypeError, ValueError) as e:
        print(f"Batch JIT compilation failed with error: {e}")
        assert False, "unhashable" in str(e) or "hash" in str(e)
