"""Linear Quadratic Regulator (LQR) system for testing policy gradients."""

import flax.struct
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from flax.struct import dataclass

from ..types import JaxRandomKey


@dataclass
class LQRState:
    """State for the LQR system.

    Attributes
    ----------
    x:
        State vector of shape (state_dim,).

    """

    x: jax.Array

    def __getitem__(self, index):
        """Allow indexing into the state vector."""
        return self.x[index]


@dataclass
class LQR:
    """Linear Quadratic Regulator system.

    Dynamics: x_{t+1} = A @ x_t + B @ u_t + w_t
    Cost: x_t^T @ Q @ x_t + u_t^T @ R @ u_t

    where w_t ~ N(0, noise_scale * I) is process noise.

    This is a standard LQR problem with known optimal solution that can be used
    to validate policy gradient implementations.

    Attributes
    ----------
    transition_matrix:
        State transition matrix of shape (state_dim, state_dim).
    control_matrix:
        Control matrix of shape (state_dim, control_dim).
    state_cost_matrix:
        State cost matrix of shape (state_dim, state_dim).
    control_cost_matrix:
        Control cost matrix of shape (control_dim, control_dim).
    discount:
        Discount factor for future costs.
    noise_scale:
        Scale of process noise.
    max_control:
        Maximum absolute control value (symmetric bounds).
    init_scale:
        Scale for random initial state sampling.

    """

    transition_matrix: jax.Array = flax.struct.field(pytree_node=False)
    control_matrix: jax.Array = flax.struct.field(pytree_node=False)
    state_cost_matrix: jax.Array = flax.struct.field(pytree_node=False)
    control_cost_matrix: jax.Array = flax.struct.field(pytree_node=False)
    discount: float = 0.99
    noise_scale: float = 0.1
    max_control: float = 1.0
    init_scale: float = 1.0

    @property
    def state_dim(self) -> int:
        """State dimension."""
        return self.transition_matrix.shape[0]

    @property
    def control_dim(self) -> int:
        """Control dimension."""
        return self.control_matrix.shape[1]

    @property
    def control_min(self) -> jax.Array:
        """Minimum control values."""
        return -self.max_control * jnp.ones(self.control_dim)

    @property
    def control_max(self) -> jax.Array:
        """Maximum control values."""
        return self.max_control * jnp.ones(self.control_dim)

    def init(self, key: JaxRandomKey) -> LQRState:
        """Initialize with random state near origin."""
        x = self.init_scale * jr.normal(key, (self.state_dim,))
        return LQRState(x=x)

    def transit(
        self, state: LQRState, control: jax.Array, key: JaxRandomKey
    ) -> LQRState:
        """Apply linear dynamics with process noise."""
        # Linear dynamics: x_{t+1} = A @ x_t + B @ u_t + noise
        x_next = (
            self.transition_matrix @ state.x
            + self.control_matrix @ control.flatten()
        )

        # Add process noise
        if self.noise_scale > 0:
            noise = self.noise_scale * jr.normal(key, x_next.shape)
            x_next = x_next + noise

        return LQRState(x=x_next)

    def cost(
        self, state: LQRState, control: jax.Array, key: JaxRandomKey
    ) -> jax.Array:
        """Quadratic cost function."""
        del key  # Unused

        # Quadratic cost: x^T Q x + u^T R u
        state_cost = state.x @ self.state_cost_matrix @ state.x
        control_cost = (
            control.flatten() @ self.control_cost_matrix @ control.flatten()
        )

        return jnp.array([state_cost + control_cost])

    def empty_control(self) -> jax.Array:
        """Return zero control of the right shape."""
        return jnp.zeros((self.control_dim,))


def visualize_2d_lqr_policy(
    lqr: LQR,
    policy_fn,
    state_range: tuple[float, float] = (-3.0, 3.0),
    resolution: int = 50,
    n_flow_samples: int = 1000,
    key: JaxRandomKey | None = None,
    figsize: tuple[float, float] = (12, 5),
):
    """Visualize policy for 2D LQR system with heatmap and flow field.

    Parameters
    ----------
    lqr:
        LQR system (must be 2D: position and velocity).
    policy_fn:
        Policy function that takes a state and returns control action.
        Should accept LQRState and return control array.
    state_range:
        Range for both position and velocity axes (min, max).
    resolution:
        Grid resolution for heatmap visualization.
    n_flow_samples:
        Number of random samples for flow field visualization.
    key:
        Random key for sampling flow field points.
    figsize:
        Figure size (width, height).

    Returns
    -------
    matplotlib.figure.Figure:
        Figure with two subplots: policy heatmap and flow field.

    """
    if key is None:
        key = jr.PRNGKey(42)

    # Verify 2D system
    if lqr.state_dim != 2:
        raise ValueError(f"LQR system must be 2D, got {lqr.state_dim}D")

    # Create grid for heatmap
    pos_range = jnp.linspace(state_range[0], state_range[1], resolution)
    vel_range = jnp.linspace(state_range[0], state_range[1], resolution)
    pos_grid, vel_grid = jnp.meshgrid(pos_range, vel_range)

    # Compute policy values on grid (vectorized)
    grid_states = jnp.stack([pos_grid.flatten(), vel_grid.flatten()], axis=1)

    def compute_policy_for_state(state_array):
        state = LQRState(x=state_array)
        control = policy_fn(state)
        return control.flatten()[0]

    policy_values_flat = jax.vmap(compute_policy_for_state)(grid_states)
    policy_grid = policy_values_flat.reshape(pos_grid.shape)

    # Sample points for flow field
    flow_states = jnp.array(
        [
            jr.uniform(
                key,
                (n_flow_samples,),
                minval=state_range[0],
                maxval=state_range[1],
            ),
            jr.uniform(
                jr.split(key)[1],
                (n_flow_samples,),
                minval=state_range[0],
                maxval=state_range[1],
            ),
        ]
    ).T

    # Compute flow field (one step dynamics)
    def compute_flow_step(state_array):
        state = LQRState(x=state_array)
        control = policy_fn(state)
        next_state = lqr.transit(
            state, control, jr.PRNGKey(0)
        )  # No noise for visualization
        return next_state.x - state.x  # Flow vector

    flow_vectors = jax.vmap(compute_flow_step)(flow_states)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Policy heatmap
    im1 = ax1.imshow(
        policy_grid,
        extent=[
            state_range[0],
            state_range[1],
            state_range[0],
            state_range[1],
        ],
        origin="lower",
        cmap="RdBu_r",
        aspect="equal",
        vmin=-1,
        vmax=1,
    )
    ax1.set_xlabel("Position")
    ax1.set_ylabel("Velocity")
    ax1.set_title("Policy Control Values")
    plt.colorbar(im1, ax=ax1, label="Control Signal")

    # Flow field
    ax2.quiver(
        flow_states[:, 0],
        flow_states[:, 1],
        flow_vectors[:, 0],
        flow_vectors[:, 1],
        alpha=0.6,
        scale_units="xy",
        scale=1,
        width=0.002,
    )
    ax2.set_xlim(state_range)
    ax2.set_ylim(state_range)
    ax2.set_xlabel("Position")
    ax2.set_ylabel("Velocity")
    ax2.set_title("Closed-Loop Flow Field")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")

    plt.tight_layout()
    return fig


def make_simple_2d_lqr(
    dt: float = 1.00,
    position_cost: float = 1.0,
    velocity_cost: float = 0.1,
    control_cost: float = 0.1,
    max_control: float = 1.0,
    discount: float = 0.99,
    noise_scale: float = 0.1,
    init_scale: float = 1.0,
) -> LQR:
    """Create a simple 2D double integrator LQR system.

    The system has state [position, velocity] and single control input.

    Parameters
    ----------
    dt:
        Time step for discrete-time dynamics.
    position_cost:
        Weight on position in the cost function.
    velocity_cost:
        Weight on velocity in the cost function.
    control_cost:
        Weight on control effort in the cost function.
    max_control:
        Maximum absolute control value.
    discount:
        Discount factor.
    noise_scale:
        Scale of process noise.
    init_scale:
        Scale for random initial state sampling.

    Returns
    -------
    LQR system configured as a 2D double integrator.

    """
    # Double integrator dynamics: [position, velocity]
    # x_{t+1} = [1 dt; 0 1] @ x_t + [0.5*dt^2; dt] @ u_t
    transition_mat = jnp.array([[1.0, dt], [0.0, 1.0]])

    control_mat = jnp.array([[0.5 * dt**2], [dt]])

    # Cost matrices
    state_cost_mat = jnp.array([[position_cost, 0.0], [0.0, velocity_cost]])

    control_cost_mat = jnp.array([[control_cost]])

    return LQR(
        transition_matrix=transition_mat,
        control_matrix=control_mat,
        state_cost_matrix=state_cost_mat,
        control_cost_matrix=control_cost_mat,
        discount=discount,
        noise_scale=noise_scale,
        max_control=max_control,
        init_scale=init_scale,
    )
