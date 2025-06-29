"""Base classes and common patterns for policy search."""

import abc
import dataclasses
import datetime
import pathlib
from typing import Any, Callable

import jax
import jax.random as jr
from flax.struct import dataclass

from ...simulate import History, create_empty_history
from ...systems.mujoco_playground import MujocoPlaygroundMDP, save_gif
from ...types import JaxRandomKey, OptimizerCarry, Policy
from ...ui import BaseCLIMetricsCallback


@dataclass
class BasePolicySearchAuxiliary:
    """Base auxiliary data class for policy search methods.

    Attributes
    ----------
    history:
        History of the simulation backing the update.
    loss:
        Last loss value of the update.

    """

    history: History
    loss: jax.Array


@dataclass
class BasePolicySearchCarry:
    """Base class for optimizer carry structures.

    Attributes
    ----------
    current:
        Current best solution.
    current_value:
        Current loss value.
    opt_carry:
        Carry of the optimizer used.
    last_history:
        Last simulation history for episode persistence.
    steps_since_init:
        Steps since last environment reset.

    """

    current: Policy
    current_value: jax.Array | None
    opt_carry: OptimizerCarry
    last_history: History | None
    steps_since_init: jax.Array


class PolicySearchCallback[Auxiliary](abc.ABC):
    """Abstract base class for policy search callbacks."""

    @abc.abstractmethod
    def __call__(
        self,
        i_update: int,
        train_history: History | None = None,
        eval_history: History | None = None,
        aux: Auxiliary | None = None,
    ) -> None:
        """Abstract method called during policy search."""

    def teardown(self):
        """Cleanup method called at end of policy search."""
        pass


type MetricExtractor[Auxiliary] = Callable[
    [History | None, History | None, Auxiliary | None], float | None
]


@dataclasses.dataclass
class BaseProgressCallback[Auxiliary](
    BaseCLIMetricsCallback,
    PolicySearchCallback[Auxiliary],
):
    """Base progress callback with configurable metric extractors."""

    metric_extractors: dict[str, MetricExtractor[Auxiliary]] = (
        dataclasses.field(default_factory=dict)
    )
    description: str = "Searching policy..."

    def __init__(
        self,
        total_steps: int,
        metric_extractors: dict[str, MetricExtractor[Auxiliary]] | None = None,
        description: str = "Searching policy...",
    ):
        """Initialize callback with dynamic metrics from extractors."""
        if metric_extractors is None:
            metric_extractors = {}
        self.metric_extractors = metric_extractors
        self.description = description

        # Initialize parent with metrics derived from extractors
        super().__init__(
            metrics=tuple(metric_extractors.keys()),
            total_steps=total_steps,
            description=description,
        )

    def __call__(
        self,
        i_update: int,
        train_history: History | None = None,
        eval_history: History | None = None,
        aux: Auxiliary | None = None,
    ) -> None:
        """Update progress bar using metric extractors."""
        metrics = {}
        for name, extractor in self.metric_extractors.items():
            try:
                value = extractor(train_history, eval_history, aux)
                metrics[name] = value
            except (AttributeError, TypeError):
                metrics[name] = None

        self._update(i_update=i_update, **metrics)


# Predefined metric extractors for common use cases
def cost_from_aux_history(
    train_history: History | None,
    eval_history: History | None,
    aux: Any | None,
) -> float | None:
    """Extract cost from aux.history.costs.mean()."""
    del train_history
    del eval_history

    if aux is not None and hasattr(aux, "history"):
        return float(aux.history.costs.mean())
    return None


def cost_from_train_history(
    train_history: History | None,
    eval_history: History | None,
    aux: Any | None,
) -> float | None:
    """Extract cost from train_history.costs.mean()."""
    del eval_history
    del aux

    if train_history is not None:
        return float(train_history.costs.mean())
    return None


def cost_from_eval_history(
    train_history: History | None,
    eval_history: History | None,
    aux: object | None,
) -> float | None:
    """Extract cost from eval_history.costs.mean()."""
    del train_history
    del aux

    if eval_history is not None:
        return float(eval_history.costs.mean())
    return None


def critic_loss_from_aux(
    train_history: History | None,
    eval_history: History | None,
    aux: Any | None,
) -> float | None:
    """Extract critic loss from aux.critic_loss.mean()."""
    del train_history
    del eval_history

    if aux is not None and hasattr(aux, "critic_loss"):
        return float(aux.critic_loss.mean())
    return None


def apx_cost_to_go_from_aux(
    train_history: History | None,
    eval_history: History | None,
    aux: Any | None,
) -> float | None:
    """Extract approximate cost-to-go from aux.apx_cost_to_go.mean()."""
    del train_history
    del eval_history

    if aux is not None and hasattr(aux, "apx_cost_to_go"):
        return float(aux.apx_cost_to_go.mean())
    return None


class BasePolicySearch(abc.ABC):
    """Abstract base class for policy search with common functionality."""

    def _create_initial_history(
        self, policy: Policy, n_simulations: int, n_steps: int
    ) -> History | None:
        """Create initial empty history if steps_per_init is used."""
        if getattr(self, "steps_per_init", None) is not None:
            keys = jr.split(jr.PRNGKey(1), n_simulations * n_steps).reshape(
                (n_simulations, n_steps, -1)
            )
            return create_empty_history(getattr(self, "mdp"), policy, keys)
        return None

    @abc.abstractmethod
    def initial_carry(self, sample_parameter: Policy) -> OptimizerCarry:
        """Initialize policy search carry state."""
        pass

    @abc.abstractmethod
    def objective(
        self,
        parameter: Policy,
        problem_data: object,
        key: JaxRandomKey,
        carry: BasePolicySearchCarry,
    ) -> tuple[jax.Array, object]:
        """Compute objective function value."""
        pass


@dataclasses.dataclass
class SaveRolloutGif[Auxiliary](PolicySearchCallback[Auxiliary]):
    """Callback to save visualisation of rollouts to a file.

    Only the first element of the history will be saved.

    Attributes
    ----------
    mdp:
        The mujoco playground mdp from which the history comes.
    filename_prefix:
        Final filename will be this preprended to "rollout-{i}.gif", where
        `i` is the number of updates done.

    """

    mdp: MujocoPlaygroundMDP
    filename_prefix: str = ""

    def __call__(
        self,
        i_update: int,
        train_history: History | None = None,
        eval_history: History | None = None,
        aux: Auxiliary | None = None,
    ) -> None:
        """Save first entry of current eval history to a file."""
        del train_history
        del aux

        if eval_history is None:
            return

        one_history = jax.tree.map(lambda leaf: leaf[0], eval_history)
        save_gif(
            self.mdp,
            one_history,
            f"{self.filename_prefix}rollout-{i_update}.gif",
        )


@dataclasses.dataclass
class WriteEvalCostsToFile[Auxiliary](PolicySearchCallback[Auxiliary]):
    """Callback for writing every evaluation cost to a file.

    Each line will be the date in isoformat, the number of updates, and the
    cost.

    Attributes
    ----------
    file_name:
        File to write costs to.

    """

    file_name: pathlib.Path

    def __call__(
        self,
        i_update: int,
        train_history: History | None = None,
        eval_history: History | None = None,
        aux: Auxiliary | None = None,
    ) -> None:
        """Write cost of history to file."""
        del train_history
        del aux

        if eval_history is None:
            return

        with self.file_name.open("a") as fp:
            now = datetime.datetime.now().isoformat()
            avg_cost = eval_history.costs.mean()
            print(f"{now} {i_update} {avg_cost:.4f}", file=fp)
