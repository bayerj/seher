"""Implementation of pendulum systems."""

import jax
import jax.numpy as jnp
import jax.random as jr
from flax.struct import dataclass

from ..types import MDP, JaxRandomKey


@dataclass
class PendulumState:
    """State of a pendulum.

    Attributes
    ----------
    angle:
        Angle of the pendulum in radians. `0.0` means facing upwards.
    velocity:
        Angular velocity.

    """

    angle: jax.Array
    velocity: jax.Array

    def cos_sin_repr(self) -> jax.Array:
        """Return cosinus/sinus representation of the state.

        The first component is the cos of the angle, the second its sine, and
        the last the velocity. This representation is locally Euclidean, which
        means that it is a better input to a neural network than the either a
        discontinuous or "wrapped" representation based on radians.

        """
        return jnp.concatenate(
            [jnp.cos(self.angle), jnp.sin(self.angle), self.velocity]
        )

    @property
    def angle_normed(self) -> jax.Array:
        """Return the normalised angle."""
        result = (
            (self.angle + jnp.pi)
            - ((self.angle + jnp.pi) // (2 * jnp.pi)) * (2 * jnp.pi)
            - jnp.pi
        )
        return result


@dataclass
class Pendulum:
    """Pendulum MDP.

    Goal is to stabilize the pendulum in the upright position. Initialisation
    is at a random angle and random velocity.

    Attributes
    ----------
    discount:
        Future costs are downweighed exponentially by this factor.
    mass:
        Mass of the pendulum.
    gravity:
        Gravity of the world.
    length:
        Length of the pendulum.
    time_diff:
        Time difference between subsequent steps.
    max_torque:
        Maximum torque that can be applied to the system.
    max_speed:
        Maximum angular velocity of the pendulum.

    """

    discount: float = 1.0
    mass: float = 1.0
    gravity: float = 10.0
    length: float = 1.0
    time_diff: float = 0.05
    max_torque: float = 2.0
    max_speed: float = 8.0

    @property
    def control_min(self) -> jax.Array:
        """Minimum control values (torque bounds)."""
        return jnp.array([-self.max_torque])

    @property
    def control_max(self) -> jax.Array:
        """Maximum control values (torque bounds)."""
        return jnp.array([self.max_torque])

    def init(self, key: JaxRandomKey) -> PendulumState:  # noqa: D102
        angle_key, velocity_key = jr.split(key)

        initial_angle = jax.random.uniform(
            angle_key, minval=-jnp.pi, maxval=jnp.pi, shape=(1,)
        )
        initial_velocity = jax.random.uniform(
            velocity_key, minval=-8, maxval=8, shape=(1,)
        )
        return PendulumState(angle=initial_angle, velocity=initial_velocity)

    init.__doc__ = MDP.init.__doc__

    def transit(  # noqa: D102
        self, state: PendulumState, control: jax.Array, key: JaxRandomKey
    ) -> PendulumState:
        del key

        control = jnp.clip(control, -self.max_torque, self.max_torque)

        angle_acc = (
            -3.0
            * self.gravity
            / (2.0 * self.length)
            * jnp.sin(state.angle + jnp.pi)
            + 3.0 / (self.mass * self.length**2.0) * control
        )
        velocity_p1 = state.velocity + self.time_diff * angle_acc
        velocity_p1 = jnp.clip(velocity_p1, -self.max_speed, self.max_speed)
        angle_p1 = state.angle + self.time_diff * velocity_p1

        result = PendulumState(angle=angle_p1, velocity=velocity_p1)

        return result

    transit.__doc__ = MDP.transit.__doc__

    def cost(  # noqa: D102
        self, state: PendulumState, control: jax.Array, key: JaxRandomKey
    ) -> jax.Array:
        del key

        result = (
            state.angle_normed**2
            + 0.1 * state.velocity**2
            + 0.001 * control**2
        )

        return result

    cost.__doc__ = MDP.cost.__doc__

    def empty_control(self) -> jax.Array:  # noqa: D102
        return jnp.empty((1,))

    empty_control.__doc__ = MDP.empty_control.__doc__


class SparsePendulum(Pendulum):
    """Variant of the pendulum which has a sparse cost.

    The cost is 0 everywhere, but -1 around the upright position.
    """

    def cost(  # noqa: D102
        self, state: PendulumState, control: jax.Array, key: JaxRandomKey
    ) -> jax.Array:
        del control, key
        return -(abs(state.angle_normed) < jnp.pi / 4).astype("float32")


class SwingupMixin:
    """Pendulum mixin for starting in resting position."""

    def init(self, key: JaxRandomKey) -> PendulumState:  # noqa: D102
        initial_angle = jnp.array([-jnp.pi])
        initial_velocity = jnp.array([0.0])
        return PendulumState(angle=initial_angle, velocity=initial_velocity)


class SparsePendulumSwingup(SparsePendulum, SwingupMixin):
    """Pendulum that has a sparse cost and starts in resting position."""

    pass


class PendulumSwingup(Pendulum, SwingupMixin):
    """Pendulum with a dense cost that starts in resting position."""

    pass
