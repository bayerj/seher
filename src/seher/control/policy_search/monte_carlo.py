"""Implementation of policy search methods."""

import abc
import functools
from typing import cast

import jax
import jax.numpy as jnp
import jax.random as jr
from flax.struct import dataclass

from ...simulate import (
    History,
    create_empty_history,
    init_or_persist,
    simulate,
)
from ...types import (
    MDP,
    JaxRandomKey,
    Policy,
    Stepper,
    StepperCarry,
)
from .base import (
    BaseProgressCallback,
    cost_from_eval_history,
    cost_from_train_history,
)


@dataclass
class MCPSAuxiliary:
    """Class for auxiliary data during Monte Carlo polic search.

    Attributes
    ----------
    history:
        History of the simulation backing the update.
    loss:
        Last loss value of the update.

    """

    history: History
    loss: jax.Array


class MCPSCallback(abc.ABC):
    """Abstract base class for callbacks during PolicySearch."""

    @abc.abstractmethod
    def __call__(
        self,
        i_update: int,
        train_history: History | None = None,
        eval_history: History | None = None,
        aux: MCPSAuxiliary | None = None,
    ) -> None:
        """Abstract method that is called during simulate."""

    def teardown(self):
        """Do nothing, no need to tear down things here."""
        pass


class MCPSCLIProgressCallback(BaseProgressCallback):
    """Unified MCPS progress callback using metric extractors."""

    def __init__(
        self, total_steps: int, description: str = "Searching policy..."
    ):
        """Initialize MCPS progress callback with predefined metrics."""
        metric_extractors = {
            "train_cost": cost_from_train_history,
            "eval_cost": cost_from_eval_history,
        }
        super().__init__(
            total_steps=total_steps,
            metric_extractors=metric_extractors,
            description=description,
        )


@dataclass
class MCPSCarry(StepperCarry[Policy]):
    """Carry for MCPS stepper."""

    current: Policy
    opt_carry: StepperCarry
    last_history: History | None
    steps_since_init: jax.Array


@dataclass
class MCPS[State, Control](Stepper[MCPSCarry, Policy, None, History]):
    """Implementation of policy search using Monte-Carlo sampling.

    Attributes
    ----------
    mdp:
        The mdp to optimise the policy for.
    policy:
        The current policy being optimised.
    n_simulations:
        The amount of parallel simulations to run per update.
    n_steps:
        Time steps to do during rollout.
    optimizer:
        Optimizer to use.
    steps_per_init:
        Number of steps to do in an mdp before it is reset.
        Must have `n_steps` as a divisor. If not given, will reset
        at every step.

    """

    mdp: MDP[State, Control, jax.Array]
    policy: Policy[State, None, Control]
    n_simulations: int
    n_steps: int
    optimizer: Stepper[StepperCarry, Policy, None, History]
    steps_per_init: int | None = None

    def _validate_steps(self):  # noqa: D105
        if self.steps_per_init is not None:
            if self.steps_per_init % self.n_steps != 0:
                raise ValueError(
                    f"{self.steps_per_init=} must be evenly divided by "
                    f"{self.n_steps=}"
                )

    def initial_carry(self, sample_parameter: Policy) -> StepperCarry:  # noqa: D102
        # TODO sample the policy somehow
        opt_carry = self.optimizer.initial_carry(sample_parameter)

        if self.steps_per_init is not None:
            keys = jr.split(
                jr.PRNGKey(1), self.n_simulations * self.n_steps
            ).reshape((self.n_simulations, self.n_steps, -1))
            last_history = create_empty_history(
                self.mdp, sample_parameter, keys
            )
        else:
            last_history = None

        return MCPSCarry(
            current=sample_parameter,
            opt_carry=opt_carry,
            last_history=last_history,
            steps_since_init=jnp.array(0.0),
        )

    initial_carry.__doc__ = Stepper.initial_carry.__doc__

    def objective(
        self,
        parameter: Policy,
        problem_data: None,
        key: JaxRandomKey,
        carry: MCPSCarry,
    ) -> tuple[jax.Array, MCPSAuxiliary]:
        """Return value of the objective of a policy on `problem_data`.

        Parameters
        ----------
        parameter
            Policy used during rollout.
        problem_data:
            Side information during these rollouts.
        key:
            RNG for all downstream stochasticity.
        carry:
            Carry from the policy search for side information, e.g. rollouts
            persistent over update steps.

        Returns
        -------
        Scalar value of the objective.

        """
        del problem_data
        policy = parameter

        @functools.partial(jax.vmap, in_axes=(None, 0, 0))
        def get_costs(policy, last_history, key):
            init_key, key = jr.split(key)

            if self.steps_per_init is not None:
                initial_state, initial_policy_carry = init_or_persist(
                    mdp=self.mdp,
                    policy=policy,
                    steps_since_init=carry.steps_since_init,
                    last_history=last_history,
                    steps_per_init=self.steps_per_init,
                    key=key,
                )

            else:
                initial_state = None
                initial_policy_carry = None

            history = simulate(
                policy=policy,
                mdp=self.mdp,
                key=key,
                n_steps=self.n_steps,
                initial_state=initial_state,
                initial_policy_carry=initial_policy_carry,
            )
            discount_factors = self.mdp.discount ** jnp.arange(
                history.costs.shape[0]
            )

            return (
                (discount_factors * history.costs).sum()
                / discount_factors.sum(),
                history,
            )

        keys = jr.split(key, self.n_simulations)
        costs, history = get_costs(policy, carry.last_history, keys)
        return costs.mean(), MCPSAuxiliary(loss=costs.mean(), history=history)

    def __call__(  # noqa: D102
        self,
        carry: StepperCarry,
        problem_data: None,
        key: JaxRandomKey,
    ) -> tuple[StepperCarry, Policy, MCPSAuxiliary]:
        self._validate_steps()
        del problem_data
        carry = cast(MCPSCarry, carry)

        objective = functools.partial(self.objective, carry=carry)
        optimizer = self.optimizer.replace(objective=objective)  # type: ignore
        opt_carry, policy, aux = optimizer(carry.opt_carry, None, key)
        carry = carry.replace(  # type: ignore
            opt_carry=opt_carry,
            current=policy,
            last_history=aux.history,
            steps_since_init=(carry.steps_since_init + self.n_steps)
            % self.steps_per_init
            if self.steps_per_init is not None
            else self.n_steps,
        )

        return carry, carry.current, aux

    __call__.__doc__ = Stepper.__call__.__doc__
