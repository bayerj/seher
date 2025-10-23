"""Tests for chunk transcription planning."""

import jax.numpy as jnp
import jax.random as jr
import optax

from seher.control.chunk_transcription_planner import ChunkTranscriptionPlanner
from seher.systems.pendulum import Pendulum


def test_one_step():
    """Test a single step of chunk transcription planning numerically."""
    key = jr.PRNGKey(42)
    mdp = Pendulum()

    chunk_size = 2
    n_plan_steps = 4
    n_iter = 5

    planner = ChunkTranscriptionPlanner(
        mdp=mdp,
        optimizer=optax.optimistic_gradient_descent(learning_rate=0.01),
        n_plan_steps=n_plan_steps,
        chunk_size=chunk_size,
        n_iter=n_iter,
        chunk_tolerance=0.1,
        initial_dual_scale=0.1,
    )

    initial_state = mdp.init(key=key)
    carry = planner.initial_carry()

    new_carry = planner(state=initial_state, carry=carry, key=key)

    assert new_carry.plan.shape == (n_plan_steps, 1)
    assert jnp.isfinite(new_carry.plan).all()


def test_warm_start():
    """Test that warm starting shifts the plan properly."""
    key = jr.PRNGKey(43)
    mdp = Pendulum()

    chunk_size = 2
    n_plan_steps = 4

    planner = ChunkTranscriptionPlanner(
        mdp=mdp,
        optimizer=optax.optimistic_gradient_descent(learning_rate=0.01),
        n_plan_steps=n_plan_steps,
        chunk_size=chunk_size,
        n_iter=2,
        warm_start=True,
    )

    initial_state = mdp.init(key=key)
    carry = planner.initial_carry()

    carry1 = planner(state=initial_state, carry=carry, key=key)
    plan1 = carry1.plan

    carry2 = planner(state=initial_state, carry=carry1, key=key)
    plan2 = carry2.plan

    assert jnp.allclose(plan1[1], plan2[0], atol=1.0)


def test_chunk_size_validation():
    """Test that planner validates chunk_size divides n_plan_steps."""
    mdp = Pendulum()

    try:
        planner = ChunkTranscriptionPlanner(
            mdp=mdp,
            optimizer=optax.optimistic_gradient_descent(learning_rate=0.01),
            n_plan_steps=5,
            chunk_size=2,
            n_iter=1,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must be divisible by" in str(e)


def test_n_chunks():
    """Test that n_chunks property is computed correctly."""
    mdp = Pendulum()

    planner = ChunkTranscriptionPlanner(
        mdp=mdp,
        optimizer=optax.optimistic_gradient_descent(learning_rate=0.01),
        n_plan_steps=20,
        chunk_size=4,
        n_iter=1,
    )

    assert planner.n_chunks == 5
