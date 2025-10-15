"""Making mujoco_playground available for seher."""

import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import mujoco_playground as mp
from flax.struct import dataclass
from jax.tree_util import tree_map
from ml_collections import ConfigDict
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from mujoco_playground._src.mjx_env import MjxEnv

from ..simulate import History
from ..types import MDP, JaxRandomKey


@dataclass
class MujocoPlaygroundMDP:
    """MDP for wrapping envs from mujoco playground.

    Attributes
    ----------
    env:
        The wrapped environment.
    discount:
        Float for discounting future costs.
    n_inner_steps:
        Amount of times the step of `.env` is applied when calling `.transit`
        a single time.

    """

    env: MjxEnv
    discount: float = 1.0
    n_inner_steps: int = 1

    @property
    def control_min(self) -> jax.Array:
        """Minimum control values from the environment."""
        control_min = jnp.where(
            self.env.mj_model.actuator_ctrllimited,
            self.env.mj_model.actuator_ctrlrange[:, 0],
            -jnp.inf,
        )
        return control_min

    @property
    def control_max(self) -> jax.Array:
        """Maximum control values from the environment."""
        control_max = jnp.where(
            self.env.mj_model.actuator_ctrllimited,
            self.env.mj_model.actuator_ctrlrange[:, 1],
            jnp.inf,
        )
        return control_max

    def init(self, key: JaxRandomKey) -> mp.State:  # noqa: D102
        return self.env.reset(key)

    init.__doc__ = MDP.init.__doc__

    def transit(  # noqa: D102
        self, state: mp.State, control: jax.Array, key: JaxRandomKey
    ) -> mp.State:
        del key
        for _ in range(self.n_inner_steps):
            state = self.env.step(state, control)

        return state

    def cost(  # noqa: D102
        self, state: mp.State, control: jax.Array, key: JaxRandomKey
    ) -> jax.Array:
        del control, key
        return jnp.array([-state.reward])

    def empty_control(self) -> jax.Array:  # noqa: D102
        return jnp.empty(self.env.action_size)

    @staticmethod
    def from_registry(
        task_name: str,
        config: ConfigDict,
        config_overrides: dict[str, str | int | list[Any]],
    ) -> "MujocoPlaygroundMDP":
        """Return environment instance based on task name.

        See
        `mujoco_playground.registry.{manipulation,location,dm_control_suite}.ALL`
        for lists.
        """
        env = mp.registry.load(
            task_name, config=config, config_overrides=config_overrides
        )
        return MujocoPlaygroundMDP(env=env)


def save_gif(
    mdp: MujocoPlaygroundMDP,
    history: History,
    filename: str | pathlib.Path,
) -> None:
    """Save visualisations of MujocoPlayground history to a gif file.

    Attributes
    ----------
    mdp:
        System to visualize for.
    history:
        Single episode data to visualize.
    filename:
        File to save to.

    """
    n_steps = history.controls.shape[0] + 1
    state_list = [
        tree_map(lambda s: s[i], history.states) for i in range(n_steps)
    ]
    frames = mdp.env.render(state_list)
    clip = ImageSequenceClip(list(frames), fps=10)
    clip.write_gif(filename, fps=10, logger=None)
