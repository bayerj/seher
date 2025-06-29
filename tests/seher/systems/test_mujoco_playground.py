"""Test the seher.mjx module."""

import jax.random as jr
import pytest

from seher.systems import mujoco_playground as mp


@pytest.fixture
def task_name():
    """Fixture for task names."""
    return "CartpoleBalance"


@pytest.mark.slow
def test_init_step_cost(task_name):
    """Check if the basic methods of MujocoPlaygroundMDP work."""
    rng = jr.PRNGKey(32)
    mdp = mp.MujocoPlaygroundMDP.from_registry(task_name)

    initial_state = mdp.init(rng)
    control = mdp.empty_control()
    next_state = mdp.transit(initial_state, control, rng)
    cost = mdp.cost(next_state, control, rng)

    assert cost.shape == (1,)
