"""Implementation of chunk-based transcription planner."""

import jax
import jax.lax as jl
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from flax.struct import dataclass, field

from ..jax_util import tree_stack
from ..stepper.chunk_transcription import (
    ChunkTranscriptionOptimizer,
    ChunkTranscriptionOptimizerCarry,
    ChunkTranscriptionParams,
    constraint_violation_penalty,
    rollout_chunk,
)
from ..types import MDP, JaxRandomKey
from .mpc import MPCPlanner


@dataclass
class ChunkTranscriptionPlannerCarry:
    """Carry for ChunkTranscriptionPlanner.

    Attributes
    ----------
    optimizer_carry:
        Carry from the underlying optimizer.

    """

    optimizer_carry: ChunkTranscriptionOptimizerCarry

    @property
    def plan(self):
        """Return the current plan (control sequence)."""
        primal, _ = self.optimizer_carry.current
        return primal.controls


def flat_to_state(flat_state: jax.Array, sample_state):
    """Convert flat array back to state structure.

    Parameters
    ----------
    flat_state:
        Flattened state array.
    sample_state:
        Sample state to get the structure from.

    Returns
    -------
    Reconstructed state with same structure as sample_state.

    """
    leaves, treedef = jtu.tree_flatten(sample_state)

    reconstructed_leaves = []
    for leaf in leaves:
        size = leaf.size
        reconstructed_leaves.append(flat_state[:size].reshape(leaf.shape))
        flat_state = flat_state[size:]

    return jtu.tree_unflatten(treedef, reconstructed_leaves)


def compute_constraint_violations(
    mdp: MDP,
    primal: ChunkTranscriptionParams,
    chunk_size: int,
    n_chunks: int,
    chunk_tolerance: float,
    sample_state,
    key: JaxRandomKey,
) -> jax.Array:
    """Compute constraint violations for chunk boundaries.

    For each chunk i, we roll out the dynamics from chunk_states[i] using
    the controls for that chunk, and compute the penalized difference between
    the reached state and chunk_states[i+1].

    Parameters
    ----------
    mdp:
        MDP defining the dynamics.
    primal:
        Primal parameters containing chunk states and controls.
    chunk_size:
        Number of timesteps per chunk.
    n_chunks:
        Number of chunks.
    chunk_tolerance:
        Tolerance band for constraints.
    key:
        RNG for stochasticity.

    Returns
    -------
    violations:
        Array of shape (n_chunks, state_dim) containing the penalized
        constraint violations at each chunk boundary.

    """
    keys = jr.split(key, n_chunks)

    def compute_violation_for_chunk(i):
        """Compute violation for chunk i."""
        chunk_key = keys[i]
        initial_state_flat = primal.chunk_states[i]
        initial_state = flat_to_state(initial_state_flat, sample_state)
        controls = jl.dynamic_slice(
            primal.controls,
            (i * chunk_size, 0),
            (chunk_size, primal.controls.shape[1]),
        )
        target_state_flat = primal.chunk_states[i + 1]

        final_state, _ = rollout_chunk(mdp, initial_state, controls, chunk_key)
        final_state_leaves = jtu.tree_leaves(final_state)
        final_state_flat = jnp.concatenate(
            [leaf.reshape(-1) for leaf in final_state_leaves]
        )

        difference = final_state_flat - target_state_flat

        # Return the raw difference - the tolerance will be applied in the Lagrangian
        return difference

    violations = jax.vmap(compute_violation_for_chunk)(jnp.arange(n_chunks))
    return violations


def compute_total_cost(
    mdp: MDP,
    primal: ChunkTranscriptionParams,
    chunk_size: int,
    n_chunks: int,
    sample_state,
    key: JaxRandomKey,
) -> jax.Array:
    """Compute total cost over the entire horizon.

    Parameters
    ----------
    mdp:
        MDP defining costs.
    primal:
        Primal parameters containing chunk states and controls.
    chunk_size:
        Number of timesteps per chunk.
    n_chunks:
        Number of chunks.
    key:
        RNG for stochasticity.

    Returns
    -------
    Total cost over all chunks.

    """
    keys = jr.split(key, n_chunks)

    def cost_for_chunk(i):
        """Compute cost for chunk i."""
        chunk_key = keys[i]
        initial_state_flat = primal.chunk_states[i]
        initial_state = flat_to_state(initial_state_flat, sample_state)
        controls = jl.dynamic_slice(
            primal.controls,
            (i * chunk_size, 0),
            (chunk_size, primal.controls.shape[1]),
        )

        _, total_cost = rollout_chunk(mdp, initial_state, controls, chunk_key)
        return total_cost

    costs = jax.vmap(cost_for_chunk)(jnp.arange(n_chunks))
    return jnp.sum(costs)


@dataclass
class ChunkTranscriptionPlanner[State]:
    """Planner using chunk-based direct transcription with OGDA.

    This planner divides the time horizon into chunks and optimizes both the
    states at chunk boundaries and the controls for each timestep. Constraints
    ensure that rolling out the dynamics within each chunk leads to the next
    chunk boundary state. The optimization is performed using Optimistic
    Gradient Descent Ascent (OGDA) on an augmented Lagrangian formulation.

    Attributes
    ----------
    mdp:
        MDP to plan for.
    optimizer:
        Optax optimizer. Can be a single optimizer (applied to both primal and
        dual) or a tuple of (primal_optimizer, dual_optimizer) for separate
        optimization.
    n_plan_steps:
        Total number of timesteps to plan for.
    chunk_size:
        Number of timesteps per chunk.
    n_iter:
        Number of optimization iterations.
    chunk_tolerance:
        Tolerance band for constraint violations.
    initial_dual_scale:
        Initial scale for Lagrangian multipliers.
    warm_start:
        If True, initialize from previous plan (shifted by one step).

    """

    mdp: MDP[State, jax.Array, jax.Array]
    optimizer: jax.tree_util.Partial | object | tuple
    n_plan_steps: int = field(pytree_node=False)
    chunk_size: int = field(pytree_node=False)
    n_iter: int = field(pytree_node=False)
    chunk_tolerance: float = 0.01
    initial_dual_scale: float = 0.1
    warm_start: bool = True

    def __post_init__(self):  # noqa: D105
        if self.n_iter <= 0:
            raise ValueError("n_iter needs to be greater than 0")
        if self.n_plan_steps % self.chunk_size != 0:
            raise ValueError(
                f"n_plan_steps ({self.n_plan_steps}) must be divisible by "
                f"chunk_size ({self.chunk_size})"
            )

    @property
    def n_chunks(self) -> int:
        """Return number of chunks."""
        return self.n_plan_steps // self.chunk_size

    def initial_carry(self) -> ChunkTranscriptionPlannerCarry:  # noqa: D102
        controls = tree_stack(
            [self.mdp.empty_control() for _ in range(self.n_plan_steps)]
        )

        sample_key = jr.PRNGKey(0)
        sample_state = self.mdp.init(sample_key)
        state_leaves = jtu.tree_leaves(sample_state)
        state_dim = sum(leaf.size for leaf in state_leaves)
        chunk_states = jnp.zeros((self.n_chunks + 1, state_dim))

        primal = ChunkTranscriptionParams(
            chunk_states=chunk_states, controls=controls
        )

        dual = jnp.zeros((self.n_chunks, state_dim)) + self.initial_dual_scale

        sample_parameter = (primal, dual)

        # Check if optimizer is a tuple (primal, dual) or single optimizer
        if isinstance(self.optimizer, tuple):
            import optax as optax_module
            primal_opt, dual_opt = self.optimizer
            # Label function: map each parameter to optimizer label
            def param_labels(params):
                # params is a tuple (primal, dual)
                return ('primal', 'dual')
            combined_optimizer = optax_module.multi_transform(
                {'primal': primal_opt, 'dual': dual_opt},
                param_labels
            )
            optimizer_to_use = combined_optimizer
        else:
            optimizer_to_use = self.optimizer

        optimizer_impl = ChunkTranscriptionOptimizer(
            objective=None,
            optimizer=optimizer_to_use,
            chunk_size=self.chunk_size,
            n_chunks=self.n_chunks,
            chunk_tolerance=self.chunk_tolerance,
            initial_dual_scale=self.initial_dual_scale,
        )

        return ChunkTranscriptionPlannerCarry(
            optimizer_carry=optimizer_impl.initial_carry(sample_parameter)
        )

    initial_carry.__doc__ = MPCPlanner.initial_carry.__doc__

    def __call__(  # noqa: D102
        self,
        state: State,
        carry: ChunkTranscriptionPlannerCarry,
        key: JaxRandomKey,
    ) -> ChunkTranscriptionPlannerCarry:
        new_carry = self.initial_carry()

        if self.warm_start:
            old_primal, old_dual = carry.optimizer_carry.current
            new_primal, new_dual = new_carry.optimizer_carry.current

            shifted_controls = jnp.concatenate(
                [old_primal.controls[1:], old_primal.controls[-1:]], axis=0
            )

            shifted_chunk_states = jnp.concatenate(
                [
                    old_primal.chunk_states[1:],
                    old_primal.chunk_states[-1:],
                ],
                axis=0,
            )

            new_primal = ChunkTranscriptionParams(
                chunk_states=shifted_chunk_states,
                controls=shifted_controls,
            )

            shifted_dual = jnp.concatenate(
                [old_dual[1:], old_dual[-1:]], axis=0
            )

            new_carry = new_carry.replace(  # type: ignore
                optimizer_carry=new_carry.optimizer_carry.replace(  # type: ignore
                    current=(new_primal, shifted_dual)
                )
            )

        primal, dual = new_carry.optimizer_carry.current
        state_leaves = jtu.tree_leaves(state)
        state_flat = jnp.concatenate([leaf.reshape(-1) for leaf in state_leaves])
        primal = primal.replace(  # type: ignore
            chunk_states=primal.chunk_states.at[0].set(state_flat)
        )
        new_carry = new_carry.replace(  # type: ignore
            optimizer_carry=new_carry.optimizer_carry.replace(  # type: ignore
                current=(primal, dual)
            )
        )

        def objective(
            parameter: tuple[ChunkTranscriptionParams, jax.Array],
            problem_data: State,
            key: JaxRandomKey,
        ) -> tuple[jax.Array, None]:
            primal, dual = parameter

            cost_key, constraint_key = jr.split(key)
            total_cost = compute_total_cost(
                self.mdp, primal, self.chunk_size, self.n_chunks, problem_data, cost_key
            )

            violations = compute_constraint_violations(
                self.mdp,
                primal,
                self.chunk_size,
                self.n_chunks,
                self.chunk_tolerance,
                problem_data,
                constraint_key,
            )

            # Standard augmented Lagrangian: L = cost + dual^T * violations
            lagrangian = total_cost + jnp.sum(dual * violations)

            return lagrangian, None

        # Check if optimizer is a tuple (primal, dual) or single optimizer
        if isinstance(self.optimizer, tuple):
            import optax as optax_module
            primal_opt, dual_opt = self.optimizer
            # Label function: map each parameter to optimizer label
            def param_labels(params):
                # params is a tuple (primal, dual)
                return ('primal', 'dual')
            combined_optimizer = optax_module.multi_transform(
                {'primal': primal_opt, 'dual': dual_opt},
                param_labels
            )
            optimizer_to_use = combined_optimizer
        else:
            optimizer_to_use = self.optimizer

        optimizer_impl = ChunkTranscriptionOptimizer(
            objective=objective,
            optimizer=optimizer_to_use,
            chunk_size=self.chunk_size,
            n_chunks=self.n_chunks,
            chunk_tolerance=self.chunk_tolerance,
            initial_dual_scale=self.initial_dual_scale,
        )

        def body_fun(_, val):
            carry_val, key_val = val
            key_val, step_key = jr.split(key_val)
            new_optimizer_carry, _, _ = optimizer_impl(
                carry=carry_val,
                problem_data=state,
                key=step_key,
            )
            return (new_optimizer_carry, key_val)

        optimizer_carry, _ = jl.fori_loop(
            0,
            self.n_iter,
            body_fun,
            (new_carry.optimizer_carry, key),
        )

        return ChunkTranscriptionPlannerCarry(optimizer_carry=optimizer_carry)

    __call__.__doc__ = MPCPlanner.__call__.__doc__
