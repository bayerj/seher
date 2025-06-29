"""Module for approximation architectures."""

from typing import Callable

import jax
import jax.nn.initializers
import jax.numpy as jnp
import jax.random as jr
from flax.struct import dataclass, field

from .types import JaxRandomKey, Policy, StateCritic

Initializer = (
    jax.nn.initializers.Initializer
    | Callable[[JaxRandomKey, tuple[int, ...]], jax.Array]
)


def symlog(state: jax.Array) -> jax.Array:
    """Return array transformed element-wise with symlog."""
    result = jnp.sign(state) * jnp.log(abs(state) + 1)
    return result


@dataclass
class MLP:
    """Multilayer-perceptron as a callable.

    Attributes
    ----------
    weights:
        List of weights, one per layer.
    biases:
        List of biases, one per layer.
    activations:
        List of activations, one per layer, including the output.
    use_layernorm:
        Whether to apply layernorm to the pre-activations.
    w_init:
        Initializer for the weights.
    b_init:
        Initializer for the biases.

    """

    weights: list[jax.Array]
    biases: list[jax.Array]
    activations: list[Callable[[jax.Array], jax.Array]] = field(
        pytree_node=False
    )
    use_layernorm: bool = field(default=False, pytree_node=False)

    @classmethod
    def make(
        cls,
        inpt_size: int,
        layer_sizes: list[int],
        output_size: int,
        activations: list[Callable[[jax.Array], jax.Array]],
        key: JaxRandomKey,
        w_init: Initializer = lambda key, shape: jr.uniform(
            key=key, shape=shape, minval=-0.1, maxval=0.1
        ),
        b_init: Initializer = lambda key, shape: jr.uniform(
            key=key, shape=shape
        ),
        use_layernorm: bool = False,
    ) -> "MLP":
        """Return MLP instance with parameters drawn according to `key`."""
        if not len(layer_sizes) + 1 == len(activations):
            raise ValueError(
                f"{len(layer_sizes) + 1=} needs to be {len(activations)=}"
            )
        weights = []
        biases = []
        for in_size, out_size in zip(
            [inpt_size, *layer_sizes], [*layer_sizes, output_size]
        ):
            w_key, b_key, key = jr.split(key, 3)
            weights.append(w_init(w_key, (in_size, out_size)))
            biases.append(b_init(b_key, (out_size,)))

        return cls(
            weights=weights,
            biases=biases,
            activations=activations,
            use_layernorm=use_layernorm,
        )

    def __call__(self, inpt: jax.Array) -> jax.Array:
        """Perform forward pass of MLP."""
        output = inpt
        for i, (w, b, act) in enumerate(
            zip(self.weights, self.biases, self.activations)
        ):
            if self.use_layernorm and not i == len(self.weights) - 1:
                output -= output.mean()
                output /= output.std() + 1e-4
            output = act(output @ w + b)

        return output


@dataclass
class StaticMLPPolicy[Observation, Control]:
    """Adapt an MLP to be a static policy for an MDP or so.

    Attributes
    ----------
    mlp:
        Function approximator to use.
    obs_to_array:
        Turn the observation that a policy gets into an array so that it can be
        given to an MLP.
    array_to_control:
        Turn the array that is output by the MLP into a control.

    """

    mlp: MLP
    obs_to_array: Callable[[Observation], jax.Array] = field(pytree_node=False)
    array_to_control: Callable[[jax.Array], Control] = field(pytree_node=False)

    def initial_carry(self) -> None:  # noqa: D102
        return None

    initial_carry.__doc__ = Policy.initial_carry.__doc__

    def __call__(  # noqa: D102
        self,
        carry: None,
        obs: Observation,
        control: Control,
        key: JaxRandomKey,
    ) -> tuple[None, Control]:
        del carry, key
        # TODO make control an input to mlp as well.
        inpt_arr = self.obs_to_array(obs)
        output_arr = self.mlp(inpt_arr)
        control = self.array_to_control(output_arr)
        return None, control


@dataclass
class StaticMLPCritic[State]:
    """Adapt an MLP to be a static critic.

    Attributes
    ----------
    mlp:
        Function approximator to use.
    state_to_array:
        Turn the state that a policy gets into an array so that it can be
        given to an MLP.

    """

    mlp: MLP
    state_to_array: Callable[[State], jax.Array] = field(pytree_node=False)

    def __call__(  # noqa: D102
        self,
        state: State,
        key: JaxRandomKey,
    ) -> jax.Array:
        del key
        inpt_arr = self.state_to_array(state)
        result_arr = self.mlp(inpt_arr)

        return result_arr

    __call__.__doc__ = StateCritic.__call__.__doc__
