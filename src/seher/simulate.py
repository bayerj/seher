"""Module for simulating systems."""

import abc
import dataclasses
import functools

import jax
import jax.experimental
import jax.lax as jl
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from flax.struct import dataclass

from seher.types import (
    MDP,
    Control,
    Cost,
    JaxRandomKey,
    Policy,
    PolicyCarry,
    State,
)
from seher.ui import BaseCLIMetricsCallback


class SimulateCallback[State, PolicyCarry, Control](abc.ABC):
    """Abstract base class for implementing callbacks for simulation.

    During a call to `simulate`, a sub class callback will be called after each
    step with `jax.experimental.io_callback`.

    """

    @abc.abstractmethod
    def __call__(
        self,
        i_step: int,
        state: State,
        policy_carry: PolicyCarry,
        control: Control,
        cost: jax.Array,
    ) -> None:
        """Abstract method that is called during simulate."""
        ...

    def teardown(self) -> None:
        """Do nothing, only fulfill the interface."""
        pass


@dataclasses.dataclass
class RichProgressCallback(
    BaseCLIMetricsCallback,
    SimulateCallback[State, PolicyCarry, Control],
):
    """Class for rich visualization of simulation progress.

    Attributes
    ----------
    total_steps:
        Number of steps this visualisation does.
    description:
        Description to show during the progress printing. Useful if you have
        multiple simulations running consecutively.

    """

    description: str = "Simulating..."
    metrics: tuple[str, ...] = dataclasses.field(
        init=False,
        default_factory=lambda: ("cost",),
    )

    def __call__(
        self,
        i_step: int,
        state: State,
        policy_carry: PolicyCarry,
        control: Control,
        cost: jax.Array,
    ) -> None:
        """Report current simulation state to the command line.

        The state, control and policy carry are ignored. The mean of the cost
        is reported.
        """
        del state, policy_carry, control
        self._update(cost=float(cost.mean()), i_update=i_step)


@dataclass
class History[State, Control, Cost, PolicyCarry]:
    """Value class for storing simulation information.

    Attributes
    ----------
    states:
        Of type `State`, but each entry has a prefix shape of `(T+1,)`,
        indexing the time steps.
    controls:
        Of type `Control`, but each entry has a prefix shape of `(T,)`,
        indexing the time steps.
    costs:
        Of type `Cost`, but each entry has a prefix shape of `(T,)`,
        indexing the time steps.
    policy_carries:
        Of type `PolicyCarry`, but each entry has a prefix shape of `(T,)`,
        indexing the time steps.

    """

    states: State
    controls: Control
    costs: Cost
    policy_carries: PolicyCarry


def simulate(
    mdp: MDP[State, Control, Cost],
    policy: Policy[State, PolicyCarry, Control],
    n_steps: int,
    key: JaxRandomKey,
    initial_state: State | None = None,
    initial_policy_carry: PolicyCarry | None = None,
    callback: SimulateCallback[State, PolicyCarry, Control] | None = None,
    jit_policy: bool = True,
    jit_init: bool = True,
    jit_transit: bool = True,
    jit_cost: bool = True,
) -> History[State, Control, Cost, PolicyCarry]:
    """Return history of rolling out `policy` on `mdp`.

    Parameters
    ----------
    mdp:
        System to rollout on.
    policy:
        Actor from which the controls come.
    n_steps:
        Number of time steps to simulate for.
    key:
        RNG for all downstream stochasticity.
    initial_state:
        States to start simulation from. If not given, draw from `mdp.init`.
    initial_policy_carry:
        Policy carries to start policies from. If not given, use the default
        initial policy carry.
    callback:
        Will be called after each iteration.
    jit_policy:
        Whether to jit the policy.
    jit_init:
        Whether to jit `mdp.init`.
    jit_transit:
        Whether to jit `mdp.transit`.
    jit_cost:
        Whether to jit `mdp.cost`.

    Returns
    -------
    History instance populated with results.

    """
    call_policy = jax.jit(policy.__call__) if jit_policy else policy
    call_init = jax.jit(mdp.init) if jit_init else mdp.init
    call_transit = jax.jit(mdp.transit) if jit_transit else mdp.transit
    call_cost = jax.jit(mdp.cost) if jit_cost else mdp.cost

    init_key, key = jr.split(key)

    if initial_state is None:
        initial_state = call_init(key=init_key)

    if initial_policy_carry is None:
        initial_policy_carry = policy.initial_carry()

    initial_control = mdp.empty_control()

    def scan_step(carry, _):
        i_step, state, policy_carry, control, key = carry

        policy_key, transit_key, cost_key, key = jr.split(key, 4)

        policy_carry, control = call_policy(
            carry=policy_carry,
            obs=state,
            control=control,
            key=policy_key,
        )

        cost = call_cost(state=state, control=control, key=cost_key)
        state_p1 = call_transit(state=state, control=control, key=transit_key)

        if callback is not None:
            jax.experimental.io_callback(
                callback, None, i_step, state, policy_carry, control, cost
            )

        new_carry = (i_step + 1, state_p1, policy_carry, control, key)
        outputs = (state_p1, control, cost, policy_carry)

        return new_carry, outputs

    initial_carry = (
        0,
        initial_state,
        initial_policy_carry,
        initial_control,
        key,
    )

    _, (states, controls, costs, policy_carries) = jax.lax.scan(
        scan_step,
        initial_carry,
        None,  # Not needed since we're just iterating n_steps times.
        length=n_steps,
    )

    result = History(
        states=states,
        controls=controls,
        costs=costs,
        policy_carries=policy_carries,
    )

    if callback is not None:
        jax.experimental.io_callback(callback.teardown, None)

    return result


def init_or_persist(
    mdp: MDP[State, Control, Cost],
    policy: Policy[State, PolicyCarry, Control],
    last_history: History[State, Control, Cost, PolicyCarry] | None,
    steps_since_init: jax.Array,
    steps_per_init: int,
    key: JaxRandomKey,
) -> tuple[State, PolicyCarry]:
    """Initialize new episode or persist from last history."""
    key, init_key = jr.split(key)

    def init(key):
        return mdp.init(key), policy.initial_carry()

    def persist(key):
        if last_history is None:
            raise ValueError(".last_history must be set")
        states = jt.map(lambda x: x[-1], last_history.states)
        policy_carries = jt.map(lambda x: x[-1], last_history.policy_carries)

        return states, policy_carries

    initial_state, initial_policy_carry = jl.cond(
        (steps_since_init % steps_per_init != 0),
        persist,
        init,
        init_key,
    )

    return initial_state, initial_policy_carry


@functools.partial(jax.vmap, in_axes=(None, None, 0))
@functools.partial(jax.vmap, in_axes=(None, None, 0))
def create_empty_history(
    mdp: MDP[State, Control, Cost],
    policy: Policy[State, PolicyCarry, Control],
    key: JaxRandomKey,
) -> History[State, Control, Cost, PolicyCarry]:
    """Create empty history structure for initialization."""
    # Create a zero-like cost structure by running cost once and mapping to
    # zeros.
    temp_state = mdp.init(key)
    temp_control = mdp.empty_control()
    temp_cost = mdp.cost(temp_state, temp_control, key)
    zero_cost = jax.tree_util.tree_map(jnp.zeros_like, temp_cost)

    return History(
        states=mdp.init(key),
        controls=mdp.empty_control(),
        policy_carries=policy.initial_carry(),
        costs=zero_cost,
    )


@functools.partial(jax.vmap, in_axes=(None, None, 0, None, None, None, 0))
def batch_simulate(
    mdp: MDP[State, Control, Cost],
    policy: Policy[State, PolicyCarry, Control],
    key: JaxRandomKey,
    n_steps: int,
    steps_since_init: jax.Array,
    steps_per_init: int | None,
    last_history: History[State, Control, Cost, PolicyCarry] | None,
) -> History[State, Control, Cost, PolicyCarry]:
    """Simulate policy on MDP with batching and episode persistence."""
    if steps_per_init is not None:
        initial_state, initial_policy_carry = init_or_persist(
            mdp=mdp,
            policy=policy,
            last_history=last_history,
            steps_per_init=steps_per_init,
            steps_since_init=steps_since_init,
            key=key,
        )

    else:
        initial_state = None
        initial_policy_carry = None

    history = simulate(
        policy=policy,
        mdp=mdp,
        key=key,
        n_steps=n_steps,
        initial_state=initial_state,
        initial_policy_carry=initial_policy_carry,
    )
    return history
