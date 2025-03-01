"""Basic types and protocls for seher."""

from typing import Protocol

from jax import Array

JaxRandomKey = Array


class MDP[State, Control, Cost](Protocol):
    def init(self, key: JaxRandomKey) -> State: ...

    def transit(
        self, state: State, control: Control, key: JaxRandomKey
    ) -> State: ...

    def cost(
        self, state: State, control: Control, key: JaxRandomKey
    ) -> Cost: ...


class CMDP[State, Control, Cost, Constraint](
    MDP[State, Control, Cost], Protocol
):
    def constraint(
        self, state: State, control: Control, key: JaxRandomKey
    ) -> Constraint: ...


class SSM[State, Control, Observation](Protocol):
    def init(self, key: JaxRandomKey) -> State: ...

    def transit(
        self, state: State, control: Control, key: JaxRandomKey
    ) -> State: ...

    def emit(
        self, state: State, control: Control, key: JaxRandomKey
    ) -> Observation: ...


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
    pass


class CPOMDP[State, Control, Observation, Cost, Constraint](
    SSM[State, Control, Observation],
    CMDP[State, Control, Cost, Constraint],
    Protocol,
):
    pass
