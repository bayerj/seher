"""Methods for learning with actor critic."""

import abc
import functools
from typing import cast

import jax
import jax.lax as jl
import jax.numpy as jnp
import jax.random as jr
import optax
from flax.struct import dataclass, field

from ...simulate import History, batch_simulate, create_empty_history
from ...types import (
    MDP,
    JaxRandomKey,
    Optimizer,
    OptimizerCarry,
    Policy,
    State,
    StateCritic,
)
from ..td_lambda import td_lambda_return
from .base import (
    BaseProgressCallback,
    apx_cost_to_go_from_aux,
    cost_from_aux_history,
    critic_loss_from_aux,
)


@functools.partial(jax.vmap, in_axes=(None, 0, None))
@functools.partial(jax.vmap, in_axes=(None, 0, None))
def _batched_critic(
    critic: StateCritic[State], state: State, key: JaxRandomKey
) -> jax.Array:
    """Apply `critic` to `state` of shape `N, T, D`."""
    return critic(state, key)


@functools.partial(jax.vmap, in_axes=(0, 0, None, None))
@functools.partial(jax.vmap, in_axes=(0, 0, None, None))
def _batched_td_lambda_return(
    costs: jax.Array, cost_to_gos: jax.Array, discount: float, td_lambda: float
):
    """Return TD-lambda returns given `costs` and `cost_to_gos`.

    Parameters
    ----------
    costs:
        Shape `(N, T, D)`.
    cost_to_gos:
        Shape `(N, T, D)`.
    discount:
        Discount of the problem.
    td_lambda:
        TD-lambda value for the mixing.

    Returns
    -------
    Shape `(N, T, D)`.

    """
    return td_lambda_return(
        costs=costs,
        cost_to_gos=cost_to_gos,
        discount=discount,
        td_lambda=td_lambda,
    )


@dataclass
class ActorCriticAuxiliary:
    """Class for auxiliary data during actor-critic optimization.

    Attributes
    ----------
    history:
        History of the simulation backing the update.
    loss:
        Last loss value of the update.
    apx_cost_to_go:
        Approximate cost to go values as predicted by the critic.
    critic_loss:
        Loss of training the critic only.

    """

    history: History
    loss: jax.Array
    apx_cost_to_go: jax.Array
    critic_loss: jax.Array


class ActorCriticCallback(abc.ABC):
    """Abstract base class for callbacks during ActorCritic."""

    @abc.abstractmethod
    def __call__(
        self,
        train_history: History,
        eval_history: History,
        aux: ActorCriticAuxiliary,
    ) -> None:
        """Abstract method that is called during simulate."""


class ActorCriticCLIProgressCallback(BaseProgressCallback):
    """Unified actor-critic progress callback using metric extractors."""

    def __init__(
        self, total_steps: int, description: str = "Training actor critic..."
    ):
        """Initialize ActorCritic progress callback with predefined metrics."""
        metric_extractors = {
            "cost": cost_from_aux_history,
            "critic_loss": critic_loss_from_aux,
            "apx_cost_to_go": apx_cost_to_go_from_aux,
        }
        super().__init__(
            total_steps=total_steps,
            metric_extractors=metric_extractors,
            description=description,
        )


@dataclass
class ActorCritic:
    """Parameters for performing Actor-Critic methods."""

    actor: Policy
    critic: StateCritic
    target_critic: StateCritic | None


@dataclass
class ActorCriticCarry:
    """Class for the state of ActorCriticOpimizer.

    Attributes
    ----------
    current:
        Current best solution.
    current_value:
        Current loss value.
    opt_carry:
        Carry of the optimizer used.

    """

    current: ActorCritic
    current_value: jax.Array | None
    opt_carry: OptimizerCarry
    last_history: History | None
    steps_since_init: jax.Array


@dataclass
class ActorCriticOptimizer[State, Control](
    Optimizer[
        OptimizerCarry[ActorCritic], ActorCritic, None, ActorCriticAuxiliary
    ]
):
    """Implementation of actor critic.

    Attributes
    ----------
    mdp:
        The mdp to optimise the policy for.
    n_simulations:
        The amount of parallel simulations to run per update.
    n_steps:
        Time steps to do during rollout.
    optimizer:
        Optimizer to use.
    critic_weight:
        Weight in the loss of the critic. The policy loss is weighed with
         `1.0`. Use this to balance the two.
    td_lambda:
        TD-lambda coefficient.
    steps_per_init:
        Number of steps to do in an mdp before it is reset.
        Must have `n_steps` as a divisor. If not given, will reset
        at every step.

    """

    mdp: MDP[State, Control, jax.Array]
    n_simulations: int
    n_steps: int
    optimizer: Optimizer[
        OptimizerCarry, ActorCritic, None, ActorCriticAuxiliary
    ]
    critic_weight: float = field(pytree_node=False)
    td_lambda: float = field(pytree_node=False)
    steps_per_init: int | None = None
    polyak_step_size: float | None = None

    def initial_carry(self, sample_parameter: ActorCritic) -> OptimizerCarry:  # noqa: D102
        if self.steps_per_init is not None:
            keys = jr.split(
                jr.PRNGKey(1), self.n_simulations * self.n_steps
            ).reshape((self.n_simulations, self.n_steps, -1))
            last_history = create_empty_history(
                self.mdp, sample_parameter.actor, keys
            )
        else:
            last_history = None

        opt_carry = self.optimizer.initial_carry(sample_parameter)

        return ActorCriticCarry(  # type: ignore
            current=sample_parameter,
            current_value=jnp.array(float("inf")),
            opt_carry=opt_carry,
            last_history=last_history,
            steps_since_init=jnp.array(0.0),
        )

    initial_carry.__doc__ = Optimizer.initial_carry.__doc__

    def objective(
        self,
        actor_critic: ActorCritic,
        problem_data: None,
        key: JaxRandomKey,
        carry: ActorCriticCarry,
    ) -> tuple[jax.Array, ActorCriticAuxiliary]:
        """Return actor critic loss.

        The loss is a sum of of the objective of `policy` on `problem_data` and
        how well the critic predicts the future return.

        Parameters
        ----------
        actor_critic:
            Parameters used during rollout, which is is the policy `.actor`
             and `.critic`.
        problem_data:
            Side information during these rollouts.
        key:
            RNG for all downstream stochasticity.
        carry:
            Carry from the actor critic for side information, e.g. rollouts
            persistent over update steps.

        Returns
        -------
        Scalar value of the objective.

        """
        del problem_data
        policy = actor_critic.actor
        critic = actor_critic.critic
        target_critic = (
            critic
            if actor_critic.target_critic is None
            else actor_critic.target_critic
        )

        key, simulate_key = jr.split(key, 2)
        simulate_keys = jr.split(simulate_key, self.n_simulations)

        history = batch_simulate(
            self.mdp,
            policy,
            simulate_keys,
            self.n_steps,
            carry.steps_since_init,
            self.steps_per_init,
            carry.last_history,
        )

        critic_key, target_critic_key, key = jr.split(key, 3)

        critic_predictions = _batched_critic(
            critic, history.states, critic_key
        )

        target_critic_predictions = _batched_critic(
            jl.stop_gradient(target_critic),
            history.states,
            target_critic_key,
        )

        apx_cost_to_gos = _batched_td_lambda_return(
            history.costs,
            target_critic_predictions,
            self.mdp.discount,
            self.td_lambda,
        )

        critic_loss = abs(critic_predictions - apx_cost_to_gos).mean()
        apx_cost_to_go = apx_cost_to_gos.mean()

        loss = apx_cost_to_go + self.critic_weight * critic_loss

        return loss, ActorCriticAuxiliary(
            history=history,
            loss=loss,
            critic_loss=critic_loss,
            apx_cost_to_go=apx_cost_to_go,
        )

    def __call__(  # noqa: D102
        self,
        carry: OptimizerCarry,
        problem_data: None,
        key: JaxRandomKey,
    ) -> tuple[OptimizerCarry, ActorCritic, ActorCriticAuxiliary]:
        del problem_data
        carry = cast(ActorCriticCarry, carry)

        objective = functools.partial(self.objective, carry=carry)
        optimizer = self.optimizer.replace(objective=objective)  # type: ignore
        opt_carry, actor_critic, aux = optimizer(carry.opt_carry, None, key)

        if (
            actor_critic.target_critic is not None
            and self.polyak_step_size is not None
        ):
            new_target_critic = optax.incremental_update(
                actor_critic.critic,
                actor_critic.target_critic,
                self.polyak_step_size,
            )
        else:
            new_target_critic = None
        # TODO it is really weird that there is opt_carry.current_value and
        # carry.current. That is super confusing, especially since the latter
        # does not have an effect on how the optimisation goes. Fix it, Justin!

        actor_critic = actor_critic.replace(target_critic=new_target_critic)
        opt_carry = opt_carry.replace(current=actor_critic)

        carry = carry.replace(opt_carry=opt_carry, current=actor_critic)  # type: ignore

        return carry, carry.current, aux

    __call__.__doc__ = Optimizer.__call__.__doc__


@dataclass
class ActorCriticEnsembleOptimizer(ActorCriticOptimizer):
    """Actor-critic implementation with an ensemble of critics.

    Attributes
    ----------
    mdp:
        The mdp to optimise the policy for.
    n_simulations:
        The amount of parallel simulations to run per update.
    n_steps:
        Time steps to do during rollout.
    optimizer:
        Optimizer to use.
    critic_weight:
        Weight in the loss of the critic. The policy loss is weighed with
         `1.0`. Use this to balance the two.
    td_lambda:
        TD-lambda coefficient.
    optimism_coeff: float
        The standard deviation of the critic predictions is multiplied with
        this, and added to the critic objective. Notably, pessimism can be
        implemented by using a negative optimisim coefficient.

    """

    optimism_coeff: float = 1.0

    def objective(  # noqa: D102
        self,
        actor_critic: ActorCritic,
        problem_data: None,
        key: JaxRandomKey,
        carry: ActorCriticCarry,
    ) -> tuple[jax.Array, ActorCriticAuxiliary]:
        del problem_data

        policy = actor_critic.actor
        critic_ensemble = actor_critic.critic

        key, simulate_key = jr.split(key, 2)
        simulate_keys = jr.split(simulate_key, self.n_simulations)
        history = batch_simulate(
            self.mdp,
            policy,
            simulate_keys,
            self.n_steps,
            carry.steps_since_init,
            self.steps_per_init,
            carry.last_history,
        )

        critic_key, key = jr.split(key)

        ensemble_batched_critic = jax.vmap(
            _batched_critic, in_axes=(0, None, None)
        )

        # Have shape (n_members, n_sims, n_steps, 1).
        critic_predictions = ensemble_batched_critic(
            critic_ensemble, history.states, critic_key
        )
        critic_predictions_ng = ensemble_batched_critic(
            jl.stop_gradient(critic_ensemble), history.states, critic_key
        )

        # One more vmap for the ensemble.
        apx_cost_to_gos = jax.vmap(
            _batched_td_lambda_return, in_axes=(None, 0, None, None)
        )(
            history.costs,
            critic_predictions_ng,
            self.mdp.discount,
            self.td_lambda,
        )

        critic_loss = abs(critic_predictions - apx_cost_to_gos).mean()

        optimistic_cost_to_go = apx_cost_to_gos.mean(
            0
        ) + self.optimism_coeff * apx_cost_to_gos.std(0)

        loss = optimistic_cost_to_go.mean() + self.critic_weight * critic_loss

        return loss, ActorCriticAuxiliary(
            history=history,
            loss=loss,
            critic_loss=critic_loss,
            apx_cost_to_go=apx_cost_to_gos.mean(),
        )

    objective.__doc__ = ActorCriticOptimizer.objective.__doc__
