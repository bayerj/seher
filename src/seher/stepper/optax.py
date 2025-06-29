"""Functionality to use optax for seher steppers."""

from typing import Callable, cast

import jax
import jax.numpy as jnp
import optax
from flax.struct import dataclass, field

from ..types import JaxRandomKey, ObjectiveFunction, Optimizer, OptimizerCarry


@dataclass
class OptaxOptimizerCarry[Parameter]:
    """Represent the state of an OptaxOptimizer.

    Attributes
    ----------
    current:
        Current solution for the problem.
    opt_state:
        Internal state of the optax optimizer.

    """

    current: Parameter
    current_value: jax.Array | None
    opt_state: tuple[optax.EmptyState, ...]


@dataclass
class OptaxOptimizer[Parameter, ProblemData, Auxiliary](
    Optimizer[OptaxOptimizerCarry[Parameter], Parameter, ProblemData, None]
):
    """Optimizer implementation making use of optax optimizers.

    Attributes
    ----------
    objective:
        Scalar objective function that is to be minimized.
    optimizer:
        Optax optimizer/gradient-transformation chain that turns
        gradients into updates of the parameters.
    grad:
        A function to get the gradient from. Anything that follows the
        signature of `jax.grad`.
    value_and_grad:
        A function to get the gradient from. Anything that follows the
        signature of `jax.value_and_grad`.

    """

    objective: ObjectiveFunction[Parameter, ProblemData, Auxiliary]
    optimizer: (
        optax.GradientTransformationExtraArgs | optax.GradientTransformation
    )
    grad: Callable | None = field(pytree_node=False, default=jax.grad)
    value_and_grad: Callable | None = field(
        pytree_node=False, default=jax.value_and_grad
    )
    has_aux: bool = True

    def initial_carry(self, sample_parameter: Parameter) -> OptimizerCarry:  # noqa: D102
        """Return an initial carry for the stepper."""
        return OptaxOptimizerCarry(
            current=sample_parameter,
            current_value=jnp.array(float("inf")),
            opt_state=self.optimizer.init(sample_parameter),  # type: ignore
        )

    initial_carry.__doc__ = Optimizer.initial_carry.__doc__

    def _call_with_value_and_grad(  # noqa: D102
        self,
        carry: OptaxOptimizerCarry,
        problem_data: ProblemData,
        key: JaxRandomKey,
    ) -> tuple[jax.Array, Parameter, tuple[optax.EmptyState, ...], Auxiliary]:
        result, grads = self.value_and_grad(  # type: ignore
            self.objective, has_aux=self.has_aux
        )(carry.current, problem_data, key)

        if self.has_aux:
            value, aux = result
        else:
            value, aux = result, None

        updates, new_opt_state = self.optimizer.update(grads, carry.opt_state)
        params = optax.apply_updates(carry.current, updates)

        return value, params, new_opt_state, aux  # type: ignore

    def _call_with_grad(  # noqa: D102
        self,
        carry: OptaxOptimizerCarry,
        problem_data: ProblemData,
        key: JaxRandomKey,
    ) -> tuple[Parameter, tuple[optax.EmptyState, ...], Auxiliary]:
        result = self.grad(  # type: ignore
            self.objective, has_aux=self.has_aux
        )(carry.current, problem_data, key)

        if self.has_aux:
            grads, aux = result
        else:
            grads, aux = result, None

        updates, new_opt_state = self.optimizer.update(grads, carry.opt_state)
        params = optax.apply_updates(carry.current, updates)

        return params, new_opt_state, aux  # type: ignore

    def __call__(  # noqa: D102
        self,
        carry: OptimizerCarry,
        problem_data: ProblemData,
        key: JaxRandomKey,
    ) -> tuple[OptimizerCarry, Parameter, Auxiliary]:
        carry = cast(OptaxOptimizerCarry, carry)

        use_grad = (
            self.grad is not jax.grad
            and self.value_and_grad is jax.value_and_grad
        ) or (self.value_and_grad is None and self.grad is not None)

        if use_grad:
            params, new_opt_state, aux = self._call_with_grad(
                carry, problem_data, key
            )
            value = None
        else:
            value, params, new_opt_state, aux = self._call_with_value_and_grad(
                carry, problem_data, key
            )

        return (
            OptaxOptimizerCarry(
                current=params,
                current_value=value,
                opt_state=new_opt_state,  # type: ignore
            ),
            params,
            aux,
        )

    __call__.__doc__ = Optimizer.__call__.__doc__
