"""Implementation of an MPPI planner."""

from typing import Callable, TypeVar

import jax
import jax.random as jr
from flax.struct import dataclass, field

from ..jax_util import tree_stack
from ..types import MDP, JaxRandomKey, Stepper, StepperCarry
from .mpc import MPCPlanner, calc_costs_of_plan

GaussianMPPICarry = TypeVar("GaussianMPPICarry")


@dataclass
class StepperPlannerCarry:
    """Carry for `StepperPlanner` instances."""

    stepper_carry: StepperCarry

    @property
    def plan(self):
        """Return the current plan."""
        return self.stepper_carry.current


@dataclass
class StepperPlanner[State]:
    """Planner using a `Stepper` instance to find plans.

    Attributes
    ----------
    mdp:
        MDP to assume during planning.
    optimizer:
        Stepper to use for finding a solution.
    n_plan_steps:
        Amount of steps to plan for.
    n_iter:
        Number of iterations to perform.
    warm_start:
        If `True`, initialize the next search distribution's location parameter
        with the one from the previous iteration.
    min_scale:
        Make sure the scale of the search distribution never goes below this
        value.
    decode_plan:
        Optional argument to decode the plan into a longer one. This is useful
        for planning with `k` support controls which are then interpolated into
        a plan of length `m > k`.

    """

    mdp: MDP[State, jax.Array, jax.Array]
    optimizer: Stepper[StepperCarry, jax.Array, None, None]
    n_plan_steps: int = field(pytree_node=False)
    n_iter: int = field(pytree_node=False)
    warm_start: bool = True
    decode_plan: Callable[[jax.Array], jax.Array] = lambda x: x

    def __post_init__(self):  # noqa: D105
        if self.n_iter <= 0:
            raise ValueError("n_iter needs to be greater than 0")

    def initial_carry(self) -> StepperPlannerCarry:  # noqa: D102
        sample_parameter = tree_stack(
            [self.mdp.empty_control() for _ in range(self.n_plan_steps)]
        )
        return StepperPlannerCarry(
            self.optimizer.initial_carry(sample_parameter=sample_parameter)
        )

    initial_carry.__doc__ = MPCPlanner.initial_carry.__doc__

    def __call__(  # noqa: D102
        self,
        state: State,
        carry: StepperPlannerCarry,
        key: JaxRandomKey,
    ) -> StepperPlannerCarry:
        new_carry = self.initial_carry()

        if self.warm_start:
            new_stepper_carry = new_carry.stepper_carry.replace(  # type: ignore
                current=new_carry.stepper_carry.current.at[:-1].set(
                    carry.stepper_carry.current[1:]
                ),
            )
            new_carry = new_carry.replace(  # type: ignore
                stepper_carry=new_stepper_carry
            )

        carry = new_carry

        def objective(
            parameter: jax.Array,
            problem_data: State,
            key: JaxRandomKey,
        ) -> tuple[jax.Array, None]:
            plan = self.decode_plan(parameter)
            return calc_costs_of_plan(
                self.mdp,
                plan,
                problem_data,
                key,
            ), None

        optimizer = self.optimizer.replace(objective=objective)  # type: ignore

        def body_fun(_, val):
            carry_val, key_val = val
            key_val, step_key = jr.split(key_val)
            new_stepper_carry, _, _ = optimizer(
                carry=carry_val,
                problem_data=state,
                key=step_key,
            )
            return (new_stepper_carry, key_val)

        stepper_carry, _ = jax.lax.fori_loop(
            0,  # Lower bound (inclusive)
            self.n_iter,  # Upper bound (exclusive)
            body_fun,  # Body function
            (carry.stepper_carry, key),  # Initial values
        )

        # Pyright complains about stepper_carry potentially being unbound
        # below. We hence ignore that line in the type checker.
        return StepperPlannerCarry(stepper_carry=stepper_carry)  # type: ignore

    __call__.__doc__ = MPCPlanner.__doc__
