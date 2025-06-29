"""Implementation of model-predictive control."""

import functools
from typing import Callable, Protocol

import jax
import jax.numpy as jnp
from flax.struct import dataclass, field
from jax.tree import map as tree_map

from ..jax_util import tree_stack
from ..simulate import simulate
from ..types import MDP, JaxRandomKey, Policy, State
from .basic_policies import OpenLoopPolicy


@functools.partial(jax.jit, static_argnames=("mdp",))
def calc_costs_of_plan(
    mdp: MDP[State, jax.Array, jax.Array],
    plan: jax.Array,
    initial_state: State,
    key: JaxRandomKey,
) -> jax.Array:
    """Return the sum of the costs a plan induces on a system.

    Parameters
    ----------
    mdp:
        System to evaluate on.
    initial_state:
        Start evaluation from this state.
    plan:
        Plan to execute.
    key:
        RNG for all downstream stochasticity.

    Returns
    -------
    Scalar cost.

    """
    n_plan_steps = plan.shape[0]
    open_loop_policy = OpenLoopPolicy[State](plan=plan)
    history = simulate(
        mdp=mdp,
        policy=open_loop_policy,
        n_steps=n_plan_steps,
        key=key,
        initial_state=initial_state,
    )
    total_cost = history.costs.sum()
    return total_cost


@dataclass
class MPCCarry[Control]:
    """Carry for `MPCPolicy` instances.

    Attributes
    ----------
    plan:
        `Control` which is a PyTree. Each leaf is prefixed in its shape by
        `(T,)`, which is the length of the plan.

    """

    plan: Control


class MPCPlanner[State, Control, Cost](Protocol):
    """Protocol for planners for `MPCPolicy`.

    Attributes
    ----------
    mdp:
        Problem to plan for.
    n_plan_steps:
        Steps to plan for.

    """

    mdp: MDP[State, Control, Cost]
    n_plan_steps: int = field(pytree_node=False)

    def initial_carry(self) -> MPCCarry[Control]:
        """Return carry for the first time step of applying the policy."""
        ...

    def __call__(
        self,
        state: State,
        carry: MPCCarry[Control],
        key: JaxRandomKey,
    ) -> MPCCarry[Control]:
        """Return the new carry of the planner.

        Parameters
        ----------
        state:
            Current state the MDP is in.
        carry:
            Carry from the last call.
        key:
            RNG for all downstream stochasticity.

        Returns
        -------
        Latest carry.

        """
        ...


@dataclass
class RandomSearchPlanner[State]:
    """Planner that just uses random trials for finding the best plan.

    Attributes
    ----------
    mdp:
        Problem to plan on.
    n_plan_steps:
        Amount of steps to plan into the future.
    n_candidates:
        Number of random candidates for plans to try.
    proposal:
        Callable that is used to generate the random candidates,
        accepting `n_candidates`, `n_plan_steps` and `control_dim`.

    """

    mdp: MDP[State, jax.Array, jax.Array]
    n_plan_steps: int = field(pytree_node=False)
    n_candidates: int = field(pytree_node=False)
    proposal: Callable[[int, int, int, JaxRandomKey], jax.Array]

    def initial_carry(self) -> MPCCarry[jax.Array]:  # noqa: D102
        plan = tree_stack(
            [self.mdp.empty_control() for _ in range(self.n_plan_steps)]
        )
        return MPCCarry(plan=plan)

    initial_carry.__doc__ = MPCPlanner.initial_carry.__doc__

    def __call__(  # noqa: D102
        self, state: State, carry: MPCCarry[jax.Array], key: JaxRandomKey
    ) -> MPCCarry[jax.Array]:
        del carry
        control_dim = self.mdp.empty_control().shape[0]
        candidates = self.proposal(
            self.n_candidates, self.n_plan_steps, control_dim, key
        )

        get_costs = jax.vmap(calc_costs_of_plan, in_axes=(None, 0, None, None))
        costs = get_costs(self.mdp, candidates, state, key)
        best_idx = jnp.argmin(costs)
        return MPCCarry(plan=candidates[best_idx])

    __call__.__doc__ = MPCPlanner.__call__.__doc__


@dataclass
class MPCPolicy[State, Control, Cost]:
    """Policy implementing model predictive control.

    Attributes
    ----------
    mdp:
        Problem to control.
    planner:
        Planner to use at each step.

    """

    mdp: MDP[State, Control, Cost]
    planner: MPCPlanner[State, Control, Cost]

    def initial_carry(self) -> MPCCarry[Control]:  # noqa: D102
        return self.planner.initial_carry()

    initial_carry.__doc__ = Policy.initial_carry.__doc__

    def __call__(  # noqa: D102
        self,
        carry: MPCCarry[Control],
        obs: State,
        control: Control,
        key: JaxRandomKey,
    ) -> tuple[MPCCarry, Control]:
        carry = self.planner(state=obs, carry=carry, key=key)
        control = tree_map(lambda x: x[0], carry.plan)
        return carry, control

    __call__.__doc__ = Policy.__call__.__doc__
