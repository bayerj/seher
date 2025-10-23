"""Implementation of chunk-based direct transcription optimizer using OGDA."""

from typing import cast

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from flax.struct import dataclass, field

from ..control.basic_policies import OpenLoopPolicy
from ..grad_util import greater_than_ste
from ..jax_util import tree_stack
from ..simulate import simulate
from ..types import (
    JaxRandomKey,
    ObjectiveFunction,
    Optimizer,
    OptimizerCarry,
)


def constraint_violation_penalty(
    difference: jax.Array, epsilon: float
) -> jax.Array:
    """Compute penalty for constraint violations with epsilon band.

    Returns a penalty that is 0 when |difference| <= epsilon, and grows
    quadratically beyond that. Uses straight-through estimation for gradients.

    Parameters
    ----------
    difference:
        Difference vector (e.g., state_predicted - state_target).
    epsilon:
        Band size. Violations are only penalized when |difference| > epsilon.

    Returns
    -------
    Scalar penalty value.

    """
    norm = jnp.linalg.norm(difference)
    violation = norm - epsilon
    mask = greater_than_ste(violation, 0.0)
    penalty = mask * violation**2
    return penalty


def rollout_chunk(
    mdp,
    initial_state: jax.Array,
    controls: jax.Array,
    key: JaxRandomKey,
) -> tuple[jax.Array, jax.Array]:
    """Roll out dynamics for a single chunk using simulate.

    Parameters
    ----------
    mdp:
        MDP to roll out.
    initial_state:
        Starting state for the chunk.
    controls:
        Control sequence for the chunk. Shape: (chunk_size, control_dim).
    key:
        RNG for stochasticity.

    Returns
    -------
    final_state:
        State reached at end of chunk.
    total_cost:
        Sum of costs incurred during chunk.

    """
    chunk_size = controls.shape[0]
    policy = OpenLoopPolicy(plan=controls)

    history = simulate(
        mdp=mdp,
        policy=policy,
        n_steps=chunk_size,
        key=key,
        initial_state=initial_state,
        initial_policy_carry=policy.initial_carry(),
    )

    final_state = jax.tree.map(lambda x: x[-1], history.states)
    total_cost = jnp.sum(history.costs)

    return final_state, total_cost


@dataclass
class ChunkTranscriptionParams:
    """Parameters for chunk transcription optimization.

    Attributes
    ----------
    chunk_states:
        States at chunk boundaries. Shape: (n_chunks + 1, state_dim).
    controls:
        Controls for all timesteps. Shape: (n_plan_steps, control_dim).

    """

    chunk_states: jax.Array
    controls: jax.Array


@dataclass
class ChunkTranscriptionOptimizerCarry:
    """Carry for ChunkTranscriptionOptimizer.

    Attributes
    ----------
    current:
        Current solution as [primal, dual] where primal contains the
        chunk_states and controls, and dual contains Lagrangian multipliers.
    current_value:
        Current value of the objective function.
    opt_state:
        Internal state of the optax optimizer.

    """

    current: tuple[ChunkTranscriptionParams, jax.Array]
    current_value: jax.Array | None
    opt_state: tuple[optax.EmptyState, ...]


@dataclass
class ChunkTranscriptionOptimizer[ProblemData](
    Optimizer[
        ChunkTranscriptionOptimizerCarry,
        tuple[ChunkTranscriptionParams, jax.Array],
        ProblemData,
        None,
    ]
):
    """Optimizer using chunk-based transcription with OGDA.

    This optimizer combines direct transcription (optimizing states) with
    direct shooting (rolling out dynamics) by dividing the time horizon into
    chunks. It uses Optimistic Gradient Descent Ascent (OGDA) to solve the
    constrained optimization problem via an augmented Lagrangian formulation.

    The primal variables are the states at chunk boundaries and all controls.
    The dual variables are Lagrangian multipliers enforcing that chunk
    boundaries match when rolling out the dynamics within each chunk.

    Attributes
    ----------
    objective:
        Objective function to minimize (typically sum of costs).
    optimizer:
        Optax optimizer to use. Should be an OGDA variant like
        optax.optimistic_gradient_descent or optax.optimistic_adam.
    chunk_size:
        Number of timesteps per chunk.
    n_chunks:
        Number of chunks to divide the horizon into.
    chunk_tolerance:
        Band size for constraint violations. Violations are only penalized
        when the difference between chunk boundaries exceeds this value.
    initial_dual_scale:
        Initial scale for Lagrangian multipliers.

    """

    objective: ObjectiveFunction | None
    optimizer: (
        optax.GradientTransformationExtraArgs | optax.GradientTransformation
    )
    chunk_size: int = field(pytree_node=False)
    n_chunks: int = field(pytree_node=False)
    chunk_tolerance: float = 0.01
    initial_dual_scale: float = 0.1

    def initial_carry(  # noqa: D102
        self, sample_parameter: tuple[ChunkTranscriptionParams, jax.Array]
    ) -> ChunkTranscriptionOptimizerCarry:
        return ChunkTranscriptionOptimizerCarry(
            current=sample_parameter,
            current_value=jnp.array(float("inf")),
            opt_state=self.optimizer.init(sample_parameter),  # type: ignore
        )

    initial_carry.__doc__ = Optimizer.initial_carry.__doc__

    def __call__(  # noqa: D102
        self,
        carry: OptimizerCarry,
        problem_data: ProblemData,
        key: JaxRandomKey,
    ) -> tuple[
        OptimizerCarry,
        tuple[ChunkTranscriptionParams, jax.Array],
        None,
    ]:
        carry = cast(ChunkTranscriptionOptimizerCarry, carry)

        if self.objective is None:
            raise ValueError("set objective first")

        # Compute value and gradients.
        (value, _), grads = jax.value_and_grad(self.objective, has_aux=True)(
            carry.current, problem_data, key
        )

        # Negate dual gradients for gradient ascent.
        primal_grads, dual_grads = grads
        grads = (primal_grads, -dual_grads)

        # Update parameters.
        updates, new_opt_state = self.optimizer.update(
            grads, carry.opt_state, carry.current
        )
        params = optax.apply_updates(carry.current, updates)

        return (
            ChunkTranscriptionOptimizerCarry(
                current=params,
                current_value=value,
                opt_state=new_opt_state,  # type: ignore
            ),
            params,
            None,
        )

    __call__.__doc__ = Optimizer.__call__.__doc__
