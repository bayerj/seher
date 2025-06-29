"""Basic types and protocls for seher."""

from typing import Protocol, TypeVar

import jax

JaxRandomKey = jax.Array

Auxiliary = TypeVar("Auxiliary")
Control = TypeVar("Control")
Cost = TypeVar("Cost")
Parameter = TypeVar("Parameter")
PolicyCarry = TypeVar("PolicyCarry")
ProblemData = TypeVar("ProblemData")
State = TypeVar("State")


class MDP[State, Control, Cost](Protocol):
    """Markov decision process.

    Attributes
    ----------
    discount:
        Rate at which future costs are discounted.
    control_min:
        Minimum values for each control dimension. Has shape `(control_dim,)`.
    control_max:
        Maximum values for each control dimension. Has shape `(control_dim,)`.

    """

    @property
    def discount(self) -> float:
        """Rate at which future costs are discounted."""
        ...

    @property
    def control_min(self) -> Control:  # noqa: D102
        """Min values for each control dimension.

        Has shape `(control_dim,)`.
        """
        ...

    @property
    def control_max(self) -> Control:  # noqa: D102
        """Max values for each control dimension.

        Has shape `(control_dim,)`.
        """
        ...

    def init(self, key: JaxRandomKey) -> State:
        """Return a random initial state from the MDP.

        Parameters
        ----------
        key:
            Jax RNG for randomness.

        Returns
        -------
        Random initial state.

        """
        ...

    def transit(
        self, state: State, control: Control, key: JaxRandomKey
    ) -> State:
        """Return the successor state of `state` given a `control`.

        Parameters
        ----------
        state:
            Current state.
        control:
            Control or action taken.
        key:
            For downstream randomness.

        Returns
        -------
        Successor state.

        """
        ...

    def cost(self, state: State, control: Control, key: JaxRandomKey) -> Cost:
        """Return the cost of using action `control` in `state`.

        Parameters
        ----------
        state:
            Current state of the MDP.
        control:
            Control applied to the MDP at its current state.
        key:
            RNG for all downstream stochasticity.

        Returns
        -------
        Cost.

        """
        ...

    def empty_control(self) -> Control:
        """Return an empty control for the MDP.

        The purpose is to carry information about the structure (PyTree,
        jax Array, ...) of the control space. Seher assumes that all
        controls are structurally equal, i.e. have the same treedef.

        """
        ...


class CMDP[State, Control, Cost, Constraint](
    MDP[State, Control, Cost], Protocol
):
    """Constrained Markov decision process."""

    def constraint(
        self, state: State, control: Control, key: JaxRandomKey
    ) -> Constraint:
        """Return constraint values when `control` is applied in `state`.

        Constraints are violated iff they are greater than `0`.

        Parameters
        ----------
        state:
            Current state.
        control:
            Control applied.
        key:
            RNG for all downstream stochasticity.

        Returns
        -------
        PyTree of all constraint values. Values greater than `0` are
        violations.

        """
        ...


class SSM[State, Control, Observation](Protocol):
    """State-space model."""

    def init(self, key: JaxRandomKey) -> State:
        """Return a random initial state from the latent state sequence.

        Parameters
        ----------
        key:
            Jax RNG for randomness.

        Returns
        -------
        Random initial state.

        """
        ...

    def transit(
        self, state: State, control: Control, key: JaxRandomKey
    ) -> State:
        """Return the successor state of `state` given a `control`.

        Parameters
        ----------
        state:
            Current state.
        control:
            Control or action taken.
        key:
            For downstream randomness.

        Returns
        -------
        Successor state.

        """
        ...

    def emit(
        self, state: State, control: Control, key: JaxRandomKey
    ) -> Observation:
        """Return an emission when applying `control` to `state`.

        Parameters
        ----------
        state:
            Current state.
        control:
            Control or action taken.
        key:
            For downstream randomness.

        Returns
        -------
        Observation.

        """
        ...


class POMDP[
    State,
    Control,
    Observation,
    Cost,
](
    SSM[State, Control, Observation],
    MDP[State, Control, Cost],
    Protocol,
):
    """Partially-observable Markov decision process.

    Combination of an MDP and an SSM.
    """

    pass


class CPOMDP[State, Control, Observation, Cost, Constraint](
    SSM[State, Control, Observation],
    CMDP[State, Control, Cost, Constraint],
    Protocol,
):
    """Constrained Partially-observable Markov decision process.

    Combination of a constrained MDP and an SSM.
    """

    pass


class Policy[Observation, Carry, Control](Protocol):
    """Policy that can be applied to sequential problems."""

    def initial_carry(self) -> Carry:
        """Return the carry for the first time step.

        Useful when initialising loops/scans.

        """
        ...

    def __call__(
        self,
        carry: Carry,
        obs: Observation,
        control: Control,
        key: JaxRandomKey,
    ) -> tuple[Carry, Control]:
        """Apply the policy to a new observation.

        Parameters
        ----------
        carry:
            Information from the policies last call, useful to implement state.
        obs:
            New observation to condition the policy on.
        control:
            Last control applied to the sequential problem--not necessarily the
            last control applied by this policy.
        key:
            Jax RNG for all downstream stochasticity.

        Returns
        -------
        Carry
            Captures the current state of the policy.
        Control
            Control to apply to the system.

        """
        ...


class StepperCarry[Parameter](Protocol):
    """Protocol for carries of steppers."""

    current: Parameter


class Stepper[StepperCarry, Parameter, ProblemData, Auxiliary](Protocol):
    """Iterative process for finding a set of parameters."""

    def initial_carry(self, sample_parameter: Parameter) -> StepperCarry:
        """Return an initial carry for the stepper."""
        ...

    def __call__(
        self,
        carry: StepperCarry,
        problem_data: ProblemData,
        key: JaxRandomKey,
    ) -> tuple[StepperCarry, Parameter, Auxiliary]:
        """Do a single step towards the solution.

        Parameters
        ----------
        carry:
            Information from the previous iteration.
        problem_data:
            Additional data reflecting the problem. Can be used to inject
            different data at each step, for example different mini batches of
            data.
        key:
            Jax RNG for all downstream stochasticity.

        Returns
        -------
        Carry
            Complete information about the current iteration, to be passed into
            the next call.
        Parameter
            Current solution.
        Auxiliary
            Auxiliary data computed during udating.

        """
        ...


class ObjectiveFunction[Parameter, ProblemData, Auxiliary](Protocol):
    """Protocol for an objective function."""

    def __call__(
        self,
        parameter: Parameter,
        problem_data: ProblemData,
        key: JaxRandomKey,
    ) -> tuple[jax.Array, Auxiliary]:
        """Return the value of the objective function and auxiliary data.

        Parameters
        ----------
        parameter:
            Parameter set to evaluate objective function for.
        problem_data:
            Side data to determine the objective, e.g. input and target data.
        key:
            RNG for all downstream stochasticity.

        Returns
        -------
        An array containing the value of the objetive function.

        Auxiliary data.

        """
        ...


class OptimizerCarry[Parameter](StepperCarry[Parameter], Protocol):
    """Protocol for carries of Optimizer instances."""

    current_value: jax.Array | None


class Optimizer[OptimizerCarry, Parameter, ProblemData, Auxiliary](
    Stepper[OptimizerCarry, Parameter, ProblemData, Auxiliary], Protocol
):
    """Iterative process for finding a set of parameters for an objective."""

    objective: ObjectiveFunction[Parameter, ProblemData, Auxiliary]

    def initial_carry(self, sample_parameter: Parameter) -> OptimizerCarry:  # noqa: D102
        ...

    initial_carry.__doc__ = Stepper.initial_carry.__doc__

    def __call__(  # noqa: D102
        self,
        carry: OptimizerCarry,
        problem_data: ProblemData,
        key: JaxRandomKey,
    ) -> tuple[OptimizerCarry, Parameter, Auxiliary]: ...

    __call__.__doc__ = Stepper.__call__.__doc__


class StateCritic[State](Protocol):
    """Critic that acts on a single state."""

    def __call__(self, state: State, key: JaxRandomKey) -> jax.Array:
        """Return an approximate cost-to-go from `state`.

        Even though we call it cost here, it can be anything else which is
        a scalar.

        Parameters
        ----------
        state:
            State of which the cost-to-go is to be returned.
        key:
            RNG for all downstream stochasticity.

        Returns
        -------
        The approximate cost-to-go.

        """
        ...
