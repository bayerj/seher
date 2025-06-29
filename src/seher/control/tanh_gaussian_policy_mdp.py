"""Augmented MDP for policy gradient methods with tanh-Gaussian policies.

This module implements an augmented MDP where stochastic policy control
sampling becomes part of the MDP transition dynamics. The policy outputs
Gaussian distribution parameters instead of concrete controls, and the MDP
handles sampling concrete controls from tanh-transformed Gaussian distributions
to ensure bounded controls.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from flax.struct import dataclass, field

from ..simulate import History
from ..types import MDP, JaxRandomKey, Policy


@dataclass
class TanhGaussianPolicyControl:
    """Control distribution parameters for tanh-Gaussian policy.

    This represents the parameters of a Gaussian distribution that will be
    transformed through tanh to produce bounded actions in [-1, 1].

    Attributes
    ----------
    loc:
        Location (mean) of the pre-tanh Gaussian distribution.
    inv_softplus_scale:
        Inverse softplus of scale to ensure positivity:
        scale = softplus(inv_softplus_scale).

    """

    loc: jax.Array
    inv_softplus_scale: jax.Array

    @property
    def scale(self) -> jax.Array:
        """Return positive scale via softplus transformation."""
        return jax.nn.softplus(self.inv_softplus_scale - 1) + 1e-4


@dataclass
class TanhGaussianPolicyState[OriginalState, ConcreteControl]:
    """Augmented state for tanh-Gaussian policy MDP.

    Attributes
    ----------
    original_state:
        State from the original MDP.
    last_control:
        Last concrete control that was sampled and applied.

    """

    original_state: OriginalState
    last_control: ConcreteControl


@dataclass
class SimulationPolicy[Observation, Carry]:
    """Policy wrapper that samples concrete actions from TanhGaussian policies.

    This wraps a policy that outputs TanhGaussianPolicyControl to sample
    concrete actions for use with the original MDP during simulation.

    Attributes
    ----------
    gaussian_policy:
        Policy that outputs TanhGaussianPolicyControl distribution parameters.
    control_min:
        Minimum values for each control dimension.
    control_max:
        Maximum values for each control dimension.

    """

    gaussian_policy: Policy[Observation, Carry, TanhGaussianPolicyControl]
    control_min: jax.Array
    control_max: jax.Array

    def initial_carry(self) -> Carry:
        """Return initial carry from the wrapped policy."""
        return self.gaussian_policy.initial_carry()

    def __call__(
        self,
        carry: Carry,
        obs: Observation,
        control: TanhGaussianPolicyControl,
        key: JaxRandomKey,
    ) -> tuple[Carry, jax.Array]:
        """Sample concrete action from TanhGaussianPolicyControl."""
        # Get TanhGaussianPolicyControl from the gaussian policy
        carry, gaussian_control = self.gaussian_policy(
            carry, obs, control, key
        )

        # Sample from Gaussian and apply tanh transformation to box
        # constraints.
        sample_key, _ = jr.split(key)
        gaussian_sample = (
            gaussian_control.loc
            + gaussian_control.scale
            * jr.normal(sample_key, gaussian_control.loc.shape)
        )
        tanh_sample = jnp.tanh(gaussian_sample)

        # Transform from [-1, 1] to [control_min, control_max].
        concrete_action = (
            self.control_min
            + (tanh_sample + 1) * (self.control_max - self.control_min) / 2
        )

        return carry, concrete_action


def _sample_concrete_control(
    control: TanhGaussianPolicyControl,
    control_min: jax.Array,
    control_max: jax.Array,
    key: JaxRandomKey,
) -> jax.Array:
    """Sample concrete control from TanhGaussianPolicyControl parameters.

    Parameters
    ----------
    control:
        Gaussian distribution parameters for control sampling.
    control_min:
        Minimum values for each control dimension.
    control_max:
        Maximum values for each control dimension.
    key:
        Random key for control sampling.

    Returns
    -------
    Concrete control values bounded in `[control_min, control_max]`.

    """
    # Sample from Gaussian and apply tanh transformation to box constraints
    gaussian_sample = control.loc + control.scale * jr.normal(
        key, control.loc.shape
    )
    tanh_sample = jnp.tanh(gaussian_sample)

    # Transform from [-1, 1] to [control_min, control_max]
    concrete_control = (
        control_min + (tanh_sample + 1) * (control_max - control_min) / 2
    )

    return concrete_control


@dataclass
class TanhGaussianPolicyMDP[OriginalState, Cost]:
    """Augmented MDP for tanh-Gaussian policy gradient methods.

    Wraps an original MDP where:
    - Original controls become part of the state (bounded concrete controls)
    - New controls are Gaussian distribution parameters
    - Transition samples from Gaussian, applies tanh, and uses original
      dynamics

    Attributes
    ----------
    original_mdp:
        The underlying MDP to wrap.

    """

    original_mdp: MDP[OriginalState, jax.Array, Cost]
    control_min: TanhGaussianPolicyControl = field(
        default=TanhGaussianPolicyControl(
            loc=jnp.array(float("-inf")),
            inv_softplus_scale=jnp.array(float("-inf")),
        )
    )
    control_max: TanhGaussianPolicyControl = field(
        default=TanhGaussianPolicyControl(
            loc=jnp.array(float("+inf")),
            inv_softplus_scale=jnp.array(float("inf")),
        )
    )

    @property
    def discount(self) -> float:
        """Return discount factor from the original MDP."""
        return self.original_mdp.discount

    def init(
        self, key: JaxRandomKey
    ) -> TanhGaussianPolicyState[OriginalState, jax.Array]:
        """Initialize augmented state with original state and zero control."""
        original_state = self.original_mdp.init(key)
        # Initialize with zero control of the right shape.
        zero_control = jax.tree.map(
            jnp.zeros_like, self.original_mdp.empty_control()
        )
        return TanhGaussianPolicyState(
            original_state=original_state,
            last_control=zero_control,
        )

    def transit(
        self,
        state: TanhGaussianPolicyState[OriginalState, jax.Array],
        control: TanhGaussianPolicyControl,
        key: JaxRandomKey,
    ) -> TanhGaussianPolicyState[OriginalState, jax.Array]:
        """Transition by sampling concrete control and applying dynamics.

        Parameters
        ----------
        state:
            Current augmented state [original_state, last_control].
        control:
            Gaussian distribution parameters for control sampling.
        key:
            Random key for control sampling and original transition.

        Returns
        -------
        New augmented state after sampling control and transitioning.

        """
        sample_key, transit_key = jr.split(key, 2)

        concrete_control = _sample_concrete_control(
            control,
            self.original_mdp.control_min,
            self.original_mdp.control_max,
            sample_key,
        )

        new_original_state = self.original_mdp.transit(
            state.original_state, concrete_control, transit_key
        )

        return TanhGaussianPolicyState(
            original_state=new_original_state,
            last_control=concrete_control,
        )

    def cost(
        self,
        state: TanhGaussianPolicyState[OriginalState, jax.Array],
        control: TanhGaussianPolicyControl,
        key: JaxRandomKey,
    ) -> Cost:
        """Return cost using the current control parameters.

        This samples a concrete control from the current
        TanhGaussianPolicyControl parameters for cost calculation. This ensures
        the cost corresponds to the current control action, not the previous
        one.

        Parameters
        ----------
        state:
            State of the augmented MDP.
        control:
            Control applied to the augmented MDP.
        key:
            Jax random key for all downstream randomness.

        Returns
        -------
        Cost for executing `control` in `state`.

        """
        sample_key, cost_key = jr.split(key, 2)

        # Sample the current control for cost calculation.
        concrete_control = _sample_concrete_control(
            control,
            self.original_mdp.control_min,
            self.original_mdp.control_max,
            sample_key,
        )

        return self.original_mdp.cost(
            state.original_state, concrete_control, cost_key
        )

    def empty_control(self) -> TanhGaussianPolicyControl:
        """Return empty Gaussian policy control for original control space."""
        original_control = self.original_mdp.empty_control()
        return TanhGaussianPolicyControl(
            loc=jax.tree.map(jnp.zeros_like, original_control),
            inv_softplus_scale=jax.tree.map(jnp.zeros_like, original_control),
        )


def compute_tanh_gaussian_policy_log_probs(
    history: History[
        TanhGaussianPolicyState, TanhGaussianPolicyControl, jax.Array, None
    ],
    control_min: jax.Array,
    control_max: jax.Array,
) -> jax.Array:
    """Compute log probabilities of controls from tanh-Gaussian policy history.

    Uses the change of variables formula for tanh transformation with box
    scaling. The transformation is:
    y = control_min + (tanh(x) + 1) * (control_max - control_min) / 2
    The Jacobian includes both tanh and scaling factors.

    Parameters
    ----------
    history:
        History from simulation with TanhGaussianPolicyMDP containing states
        and controls.
    control_min:
        Minimum values for each control dimension.
    control_max:
        Maximum values for each control dimension.

    Returns
    -------
    Log probabilities of the concrete controls under the tanh-Gaussian
    distributions. Shape: [n_simulations, n_steps-1] (one less than states
    due to transition structure).

    """
    concrete_controls = history.states.last_control[
        :, 1:
    ]  # Shape: [n_sims, T-1, control_dim]

    eps = 1e-6

    locs = history.controls.loc[:, :-1]  # Shape: [n_sims, T-1, control_dim]
    scales = (
        history.controls.scale[:, :-1] + eps
    )  # Shape: [n_sims, T-1, control_dim]

    tanh_samples = (
        2 * (concrete_controls - control_min) / (control_max - control_min) - 1
    )

    clamped_tanh = jnp.clip(tanh_samples, -1 + eps, 1 - eps)
    gaussian_samples = jnp.arctanh(clamped_tanh)

    gaussian_log_probs = -0.5 * (
        ((gaussian_samples - locs) / scales) ** 2
        + 2 * jnp.log(scales)
        + jnp.log(2 * jnp.pi)
    )

    # Apply change of variables correction: subtract log |dy/dx|.
    tanh_derivative = 1 - clamped_tanh**2
    scale_factor = (control_max - control_min) / 2
    jacobian_correction = jnp.log(tanh_derivative * scale_factor + eps)

    # Sum over control dimensions
    log_probs = (gaussian_log_probs - jacobian_correction).sum(axis=-1)

    return log_probs


def convert_tanh_gaussian_history_to_original(
    augmented_history: History[
        TanhGaussianPolicyState, TanhGaussianPolicyControl, jax.Array, None
    ],
    control_min: jax.Array,
    control_max: jax.Array,
) -> History:
    """Convert TanhGaussianPolicyMDP history back to original MDP format.

    This function undoes the augmentation strategy by:
    1. Extracting original states from TanhGaussianPolicyState.original_state.
    2. Using the concrete controls that were actually applied
      (from last_control).
    3. Keeping the same costs and policy carries.

    Parameters
    ----------
    augmented_history:
        History from simulation with TanhGaussianPolicyMDP.
    control_min:
        Minimum control values for the original MDP.
    control_max:
        Maximum control values for the original MDP.

    Returns
    -------
    History with original states and concrete controls, making the augmentation
    transparent to the user.

    """
    # Extract original states
    original_states = augmented_history.states.original_state

    # Extract concrete controls that were actually applied
    # The key insight: last_control[t] contains the control that was applied
    # during the transition from state[t-1] to state[t]
    #
    # For a history with T states, we have:
    # - states[0] is initial state
    # - states[1] is after applying controls[0] (which is last_control[1])
    # - states[2] is after applying controls[1] (which is last_control[2])
    # - ...
    # - states[T-1] is after applying controls[T-2]
    #   (which is last_control[T-1])
    #
    # So the controls sequence is last_control[1:T]
    # This gives us T-1 controls for T states, which is correct

    # Get concrete controls that were actually applied (skip the initial zero
    # control).
    concrete_controls = augmented_history.states.last_control[..., 1:, :]

    return History(
        states=original_states,
        controls=concrete_controls,
        costs=augmented_history.costs,
        policy_carries=augmented_history.policy_carries,
    )
