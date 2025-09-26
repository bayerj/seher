"""Implementation of MPPI optimizer."""

from typing import Callable, cast

import jax
import jax.lax as jl
import jax.nn
import jax.numpy as jnp
import jax.random as jr
from flax.struct import dataclass, field

from ..types import (
    JaxRandomKey,
    ObjectiveFunction,
    Stepper,
    StepperCarry,
)


@dataclass
class GaussianMPPIOptimizerCarry(StepperCarry[jax.Array]):
    """Carry for a `GaussianMPPIOptimizer` instance.

    Attributes
    ----------
    current:
        The last devised solution. Also used as the location parameter of the
        search distribution.
    scale:
        The scale parameter of the last plan.

    """

    current: jax.Array
    scale: jax.Array


@dataclass
class GaussianMPPIOptimizer[ProblemData](
    Stepper[StepperCarry[jax.Array], jax.Array, ProblemData, None]
):
    """Stepper implementation that uses MPPI with a Gaussian.

    Attributes
    ----------
    objective:
        Objective function to optimize.
    n_candidates:
        At each iteration, draw as many candidates.
    top_k:
        Calculate distribution parameters for the next iteration
    initial_loc:
        Location parameter of the initial search distributions.
    initial_scale:
        Scale param       based on the best `top_k` candidates.
    warm_start:
        If `True`, initialize the next search distribution's location parameter
        with the one from the previous iteration.
    min_scale:
        Make sure the scale of the search distribution never goes below this
        value.
    temperature:
        Each of the top candidates contributes to the new location and scale
        parameters depending on its total cost. This dependence is done by a
        "softmax". The arguments to the softmax are multiplied with
        `temperature` to control the sharpness. Higher temperatures
        result in sharper contributions, i.e. for -> oo this would be the same
        as top 1, while for 0 it is a uniform contribution.

    """

    objective: ObjectiveFunction | None
    n_candidates: int = field(pytree_node=False)
    top_k: int
    initial_loc: jax.Array
    initial_scale: jax.Array
    warm_start: bool = True
    min_scale: float = 0.1
    max_scale: float = jnp.inf
    temperature: float = 0.0
    project_candidates: Callable[[jax.Array], jax.Array] = lambda x: x

    # TODO: adapt the signature to return GaussianMPPIOptimizerCarry and
    # pyright still passes.
    def initial_carry(  # noqa: D102
        self,
        sample_parameter: jax.Array,
    ) -> StepperCarry[jax.Array]:
        initial_loc = jnp.zeros_like(sample_parameter) + self.initial_loc
        initial_scale = jnp.zeros_like(sample_parameter) + self.initial_scale

        return GaussianMPPIOptimizerCarry(
            current=initial_loc, scale=initial_scale
        )

    initial_carry.__doc__ = Stepper.initial_carry.__doc__

    # TODO: see above.
    def __call__(  # noqa: D102
        self,
        carry: StepperCarry[jax.Array],
        problem_data: ProblemData,
        key: JaxRandomKey,
    ) -> tuple[StepperCarry[jax.Array], jax.Array, None]:
        carry = cast(GaussianMPPIOptimizerCarry, carry)

        if self.objective is None:
            raise ValueError("set objective first")

        draw_key, eval_key, key = jr.split(key, 3)
        # Draw candidates.
        candidates = (
            jr.normal(
                shape=(self.n_candidates, *carry.current.shape), key=draw_key
            )
            * carry.scale[jnp.newaxis]
            + carry.current[jnp.newaxis]
        )
        candidates = self.project_candidates(candidates)
        get_costs = jax.vmap(self.objective, in_axes=(0, None, None))
        total_costs, _ = get_costs(candidates, problem_data, eval_key)

        # Pick the best k.
        _, best_idxs = jl.top_k(-total_costs, k=self.top_k)
        best = candidates[best_idxs]
        best_costs = total_costs[best_idxs]

        weights = jax.nn.softmax(-best_costs / self.temperature).reshape(
            (-1, 1, 1)
        )
        loc = (best * weights).sum(0)
        scale = (weights * (best - loc) ** 2).sum(0) ** 0.5
        scale = jnp.maximum(scale, self.min_scale)
        scale = jnp.minimum(scale, self.max_scale)

        return GaussianMPPIOptimizerCarry(current=loc, scale=scale), loc, None

    __call__.__doc__ = Stepper.__call__.__doc__
