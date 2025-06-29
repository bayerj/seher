"""Module for basic policies."""

import jax
import jax.numpy as jnp
from flax.struct import dataclass, field

from ..types import MDP, JaxRandomKey, Policy


@dataclass
class OpenLoopPolicy[State]:
    """Policy that just executes an open loop plan.

    Attributes
    ----------
    plan:
        Array containing the controls to apply to an `MDP` instance. Time steps
        are advancing along first axis.

    """

    plan: jax.Array

    def __call__(  # noqa: D102
        self,
        carry: jax.Array,
        obs: State,
        control: jax.Array,
        key: JaxRandomKey,
    ) -> tuple[jax.Array, jax.Array]:
        del obs, control, key
        return carry + 1, self.plan[carry]

    __call__.__doc__ = Policy.__call__.__doc__

    def initial_carry(self) -> jax.Array:  # noqa: D102
        return jnp.array(0)

    initial_carry.__doc__ = Policy.initial_carry.__doc__


@dataclass
class DumbPolicy[State, Control, Cost]:
    """A policy that just returns `mdp.empty_control()`.

    Attributes
    ----------
    mdp:
        MDP to get `empty_control()` from.
    parameters:
        Just s.t. there are parameters to optimise, although they don't have
        any effect.

    """

    mdp: MDP[State, Control, Cost] = field(pytree_node=False)
    parameters: jax.Array  # Just so that this is parametric.

    def initial_carry(self) -> None:  # noqa: D102
        return None

    initial_carry.__doc__ = Policy.initial_carry.__doc__

    def __call__(  # noqa: D102
        self, carry: None, obs: State, control: Control, key: JaxRandomKey
    ) -> tuple[None, Control]:
        del carry, obs, control, key
        return None, self.mdp.empty_control()

    __call__.__doc__ = Policy.__call__.__doc__
