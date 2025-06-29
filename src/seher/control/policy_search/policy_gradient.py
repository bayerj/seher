"""Policy gradient optimizer using REINFORCE with TanhGaussianPolicyMDP."""

import functools
from typing import Callable, cast

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
from ..tanh_gaussian_policy_mdp import (
    TanhGaussianPolicyMDP,
    compute_tanh_gaussian_policy_log_probs,
)


@dataclass
class PolicyGradientAuxiliary:
    """Auxiliary data for policy gradient optimization.

    Attributes
    ----------
    history:
        History of the simulation backing the update.
    loss:
        Last loss value of the update.
    returns:
        Episode returns for each simulation.
    log_probs:
        Action log probabilities for each step.

    """

    history: History
    loss: jax.Array
    returns: jax.Array
    log_probs: jax.Array


@dataclass
class PolicyGradientCarry(StepperCarry[Policy]):
    """Carry for policy gradient optimizer.

    Attributes
    ----------
    current:
        Current policy parameters.
    opt_carry:
        Optimizer carry state.
    last_history:
        Last simulation history for episode persistence.
    steps_since_init:
        Steps since last environment reset.

    """

    current: Policy
    opt_carry: StepperCarry
    last_history: History | None
    steps_since_init: jax.Array


@dataclass
class PolicyGradientOptimizer[State](
    Stepper[PolicyGradientCarry, Policy, None, PolicyGradientAuxiliary]
):
    """Policy gradient optimization using REINFORCE with TanhGaussianPolicyMDP.

    This optimizer wraps an original MDP with TanhGaussianPolicyMDP to enable
    true policy gradient methods where actions are sampled from the policy
    distribution (with tanh transformation for bounded actions) rather than
    sampling policy parameters.

    Attributes
    ----------
    mdp:
        The original MDP to optimize on (must have Array controls).
    gaussian_policy:
        Policy that outputs Gaussian distribution parameters (pre-tanh).
    n_simulations:
        Number of parallel simulations per update.
    n_steps:
        Time steps per rollout.
    optimizer:
        Gradient-based optimizer for policy parameters.
    baseline_fn:
        Optional baseline function for variance reduction.
    steps_per_init:
        Number of steps before environment reset.

    """

    mdp: MDP[State, jax.Array, jax.Array]
    gaussian_policy: Policy
    n_simulations: int
    n_steps: int
    optimizer: Stepper[StepperCarry, Policy, None, tuple[jax.Array, jax.Array]]
    baseline_fn: Callable | None = None
    steps_per_init: int | None = None

    def __post_init__(self):
        """Validate steps configuration."""
        if self.steps_per_init is not None:
            if self.steps_per_init % self.n_steps != 0:
                raise ValueError(
                    f"steps_per_init ({self.steps_per_init}) must be evenly "
                    f"divided by n_steps ({self.n_steps})"
                )

    def initial_carry(self, sample_parameter: Policy) -> PolicyGradientCarry:
        """Initialize policy gradient carry state."""
        opt_carry = self.optimizer.initial_carry(sample_parameter)

        if self.steps_per_init is not None:
            # Create TanhGaussianPolicyMDP for initial history
            augmented_mdp = TanhGaussianPolicyMDP(original_mdp=self.mdp)
            keys = jr.split(
                jr.PRNGKey(1), self.n_simulations * self.n_steps
            ).reshape((self.n_simulations, self.n_steps, -1))
            last_history = create_empty_history(
                augmented_mdp, self.gaussian_policy, keys
            )
        else:
            last_history = None

        return PolicyGradientCarry(
            current=sample_parameter,
            opt_carry=opt_carry,
            last_history=last_history,
            steps_since_init=jnp.array(0.0),
        )

    def objective(
        self,
        parameter: Policy,
        problem_data: None,
        key: JaxRandomKey,
        carry: PolicyGradientCarry,
    ) -> tuple[jax.Array, PolicyGradientAuxiliary]:
        """Compute REINFORCE objective using GaussianPolicyMDP.

        Parameters
        ----------
        parameter:
            Policy parameters to evaluate.
        problem_data:
            Unused for policy gradient.
        key:
            Random key for simulation.
        carry:
            Carry state with simulation history.

        Returns
        -------
        Negative expected return and auxiliary data.

        """
        del problem_data

        # Create the augmented MDP
        augmented_mdp = TanhGaussianPolicyMDP(original_mdp=self.mdp)

        # Update the Gaussian policy with new parameters
        updated_policy = parameter

        @functools.partial(jax.vmap, in_axes=(None, 0, 0))
        def get_episode_data(policy, last_history, key):
            if self.steps_per_init is not None:
                initial_state, initial_policy_carry = init_or_persist(
                    mdp=augmented_mdp,
                    policy=policy,
                    steps_since_init=carry.steps_since_init,
                    last_history=last_history,
                    steps_per_init=self.steps_per_init,
                    key=key,
                )
            else:
                initial_state = None
                initial_policy_carry = None

            # Simulate with the augmented MDP
            history = simulate(
                policy=policy,
                mdp=augmented_mdp,
                key=key,
                n_steps=self.n_steps,
                initial_state=initial_state,
                initial_policy_carry=initial_policy_carry,
            )

            # Compute discounted returns
            discount_factors = augmented_mdp.discount ** jnp.arange(
                history.costs.shape[0]
            )
            episode_return = (discount_factors * history.costs).sum()

            return episode_return, history

        keys = jr.split(key, self.n_simulations)
        returns, batch_history = get_episode_data(
            updated_policy, carry.last_history, keys
        )

        # Compute log probabilities from the augmented MDP history
        log_probs = compute_tanh_gaussian_policy_log_probs(
            batch_history, self.mdp.control_min, self.mdp.control_max
        )

        # Compute REINFORCE gradient estimator
        # Returns shape: [n_simulations], log_probs shape: [n_simulations, T-1]
        returns_expanded = returns[:, None]  # [n_simulations, 1]

        # Apply baseline if provided
        if self.baseline_fn is not None:
            baseline = self.baseline_fn(returns)
            returns_expanded = returns_expanded - baseline

        # Policy gradient loss: E[log π(a|s) * C] (minimizing cost)
        # Sum over time steps, mean over simulations
        pg_loss = (log_probs * returns_expanded).sum(axis=1).mean()

        auxiliary = PolicyGradientAuxiliary(
            history=batch_history,
            loss=pg_loss,
            returns=returns,
            log_probs=log_probs,
        )

        return pg_loss, auxiliary

    def __call__(
        self,
        carry: StepperCarry,
        problem_data: None,
        key: JaxRandomKey,
    ) -> tuple[StepperCarry, Policy, PolicyGradientAuxiliary]:
        """Perform one policy gradient optimization step."""
        del problem_data
        carry = cast(PolicyGradientCarry, carry)

        # Create objective function with carry closure
        objective = functools.partial(self.objective, carry=carry)
        optimizer = self.optimizer.replace(objective=objective)  # type: ignore

        # Perform optimization step
        opt_carry, policy, aux = optimizer(carry.opt_carry, None, key)

        # Update carry state
        new_carry = carry.replace(  # type: ignore
            opt_carry=opt_carry,
            current=policy,
            last_history=aux.history,
            steps_since_init=(carry.steps_since_init + self.n_steps)
            % self.steps_per_init
            if self.steps_per_init is not None
            else self.n_steps,
        )

        return new_carry, policy, aux
