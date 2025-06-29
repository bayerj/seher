"""Policy gradient optimization on LQR systems with hyperparameter tuning."""

import dataclasses
import functools
import pathlib
from typing import Callable, cast

import defopt
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import optuna
import optuna.pruners
import rich
from matplotlib import pyplot as plt

from seher.control.policy_search.base import (
    BaseProgressCallback,
    cost_from_eval_history,
)
from seher.control.solvers import (
    BaseSolver,
    PolicyGradientSolver,
    PolicySearchSolver,
    plot_simulation_costs,
)
from seher.systems.lqr import LQR, make_simple_2d_lqr, visualize_2d_lqr_policy
from seher.types import MDP


@dataclasses.dataclass
class OptunaTrialCallback:
    """Callback for optuna trial management, reporting, and pruning.

    Attributes
    ----------
    trial : optuna.Trial
        The optuna trial to report to and check for pruning.

    """

    trial: optuna.Trial

    def __call__(
        self, i_update, train_history=None, eval_history=None, **kwargs
    ):
        """Report intermediate value to the study and maybe prune."""
        del train_history
        del kwargs
        if eval_history is not None:
            avg_cost = eval_history.costs.mean()

            # Report to optuna
            self.trial.report(avg_cost, i_update)

            if not jnp.isfinite(avg_cost) or self.trial.should_prune():
                raise optuna.TrialPruned()


@dataclasses.dataclass
class PolicyVisualizationCallback:
    """Callback for visualizing policy during training.

    Attributes
    ----------
    mdp:
        The LQR MDP system.
    obs_to_array:
        Function to convert observations to arrays.
    output_dir:
        Directory to save visualization images.

    """

    mdp: MDP
    solver: BaseSolver[MDP]
    obs_to_array: Callable
    output_dir: pathlib.Path | str = "./"

    def __post_init__(self):
        """Create output directory if it doesn't exist."""
        pathlib.Path(self.output_dir).mkdir(exist_ok=True)

    def __call__(
        self,
        i_update,
        train_history=None,
        eval_history=None,
        **kwargs,
    ):
        """Visualize policy if update interval is reached."""
        del train_history, kwargs
        if eval_history is None:
            return

        def policy_wrapper(state):
            obs_array = self.obs_to_array(state)
            assert self.solver.policy is not None

            # Policy interface
            _, control = self.solver.policy(
                control=self.mdp.empty_control(),
                carry=self.solver.policy.initial_carry(),
                obs=obs_array,
                key=jr.PRNGKey(0),
            )

            return control

        policy_fig = visualize_2d_lqr_policy(
            cast(LQR, self.mdp), policy_wrapper
        )

        if eval_history is not None:
            avg_cost = eval_history.costs.mean()
            policy_fig.suptitle(
                f"Policy at Update {i_update} (Cost: {avg_cost:.3f})"
            )
        else:
            policy_fig.suptitle(f"Policy at Update {i_update}")

        filename = f"{self.output_dir}/policy_update_{i_update:06d}.png"
        policy_fig.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(policy_fig)


def obs_to_array_lqr(state):
    """Convert state to array."""
    # Handle both original state and TanhGaussianPolicyState recursively
    while hasattr(state, "original_state"):
        state = state.original_state

    # Handle LQR state
    if hasattr(state, "x"):
        return state.x
    else:
        # Fallback for other state types
        return state


def suggest_parameters(trial) -> dict:
    """Return PolicyGradientSolver-compatible parameters for optuna trial."""
    n_hidden = trial.suggest_int("policy.n_hidden", low=0, high=32)
    activation_name = trial.suggest_categorical(
        "activation", ["soft_sign", "swish", "relu", "tanh"]
    )

    # Convert activation string to actual function
    activation_fn = getattr(jnn, activation_name)

    return {
        # Core solver parameters
        "n_simulations": trial.suggest_int("n_simulations", low=4, high=128),
        "solve_seed": trial.suggest_int("solve_seed", low=0, high=1337),
        "policy_init_seed": trial.suggest_int(
            "policy_init_seed", low=0, high=1337
        ),
        # Optimizer configuration
        "optax_optimizer": trial.suggest_categorical(
            "optax_optimizer", ["adam", "sgd"]
        ),
        "optax_optimizer_kws": {
            "learning_rate": trial.suggest_float(
                "learning_rate", low=1e-5, high=1e-3, log=True
            )
        },
        "max_grad_norm": trial.suggest_float(
            "max_grad_norm", low=1.0, high=10000.0
        ),
        # Policy MLP configuration
        "policy_mlp_kws": {
            "layer_sizes": [n_hidden] if n_hidden > 0 else [],
            "activations": (
                [activation_fn, lambda x: x] if n_hidden > 0 else [lambda x: x]
            ),
        },
        # Baseline function for variance reduction
        "baseline_fn": (lambda returns: returns.mean())
        if trial.suggest_categorical("use_baseline", [True, False])
        else None,
    }


def objective(
    trial: optuna.Trial | None,
    episode_length: int = 100,
    env_steps_per_update: int = 16,
    updates_per_eval: int = 100,
    total_updates: int = 5_000,
    parameters: dict | None = None,
    suggest_parameters: Callable[[optuna.Trial], dict] = suggest_parameters,
    eval_episode_length: int | None = None,
    eval_seed: int = 1337,
    solver_type: str = "policy_gradient",
):
    """Return performance of trial or parameters using PolicyGradientSolver."""
    if parameters is None:
        if not isinstance(trial, optuna.Trial):
            raise ValueError("either parameters or trial must not be None")
        parameters = suggest_parameters(trial)

    if eval_episode_length is None:
        eval_episode_length = episode_length

    mdp = make_simple_2d_lqr(
        dt=0.5,
        position_cost=1.0,
        velocity_cost=0.0,
        control_cost=0.0,
        discount=0.99,
        noise_scale=0.0,
        init_scale=0.5,
    )

    # Not all parameters are meant to initialize the solver.
    solver_params = {
        k: v
        for k, v in parameters.items()
        if k not in ("solve_seed", "policy_init_seed")
    }

    # Choose solver based on solver_type
    if solver_type == "policy_gradient":
        solver = PolicyGradientSolver(
            obs_to_array=obs_to_array_lqr,
            episode_length=episode_length,
            steps_per_update=env_steps_per_update,
            max_updates=total_updates,
            updates_per_eval=updates_per_eval,
            eval_n_simulations=16,
            **solver_params,
        )
    elif solver_type == "monte_carlo":
        # Filter parameters relevant to PolicySearchSolver (MCPS)
        mcps_params = {
            k: solver_params[k]
            for k in [
                "n_simulations",
                "optax_optimizer",
                "optax_optimizer_kws",
                "policy_mlp_kws",
            ]
        }
        solver = PolicySearchSolver(
            obs_to_array=obs_to_array_lqr,
            episode_length=episode_length,
            steps_per_update=env_steps_per_update,
            max_updates=total_updates,
            updates_per_eval=updates_per_eval,
            eval_n_simulations=16,
            **mcps_params,
        )
    else:
        raise ValueError(
            f"Unknown solver_type: {solver_type}. "
            "Use 'policy_gradient' or 'monte_carlo'"
        )

    # Only use this callback when we are running a trial.
    if trial:
        solver.callbacks.append(OptunaTrialCallback(trial=trial))

    # Add progress callback
    progress_callback = BaseProgressCallback(
        total_steps=total_updates,
        metric_extractors={
            "eval_cost": cost_from_eval_history,
            "min_control": lambda th, eh, aux: th.controls.min()
            if th
            else None,
            "max_control": lambda th, eh, aux: th.controls.max()
            if th
            else None,
        },
        description="Policy Gradient Training on LQR...",
    )
    solver.callbacks.append(progress_callback)

    # Add policy visualization callback
    viz_callback = PolicyVisualizationCallback(
        mdp=mdp,
        solver=solver,
        obs_to_array=obs_to_array_lqr,
        output_dir=f"./{trial._trial_id}/" if trial is not None else "./",
    )
    solver.callbacks.append(viz_callback)

    eval_key = jr.PRNGKey(eval_seed)
    try:
        solver.solve(
            mdp,
            key=jr.PRNGKey(parameters["solve_seed"]),
            eval_key=eval_key,
            policy_init_key=jr.PRNGKey(parameters["policy_init_seed"]),
        )
    except optuna.TrialPruned as e:
        raise e
    finally:
        for callback in solver.callbacks:
            if hasattr(callback, "teardown"):
                callback.teardown()

    history = solver.simulate(
        n_steps=eval_episode_length, n_simulations=16, key=eval_key
    )

    plot_simulation_costs(history=history)

    final_avg_cost = float(history.costs.mean())
    rich.print(f"Final average cost: {final_avg_cost:.4f}")
    return final_avg_cost


def run_simple(
    total_updates: int = 5_000,
    *,
    solver_type: str = "policy_gradient",
):
    """Run policy search solver on LQR system with simple parameters.

    Parameters
    ----------
    total_updates:
        Number of updates to perform.
    solver_type:
        Type of solver to use ('policy_gradient' or 'monte_carlo').

    """
    # Default parameters for policy search on LQR
    parameters = {
        "n_simulations": 123,
        "optax_optimizer": "sgd",
        "optax_optimizer_kws": {"learning_rate": 0.00017119554359135567},
        "policy_mlp_kws": {
            "layer_sizes": [14],
            "activations": [jnn.tanh, lambda x: x],
        },
        "max_grad_norm": 5055.71947049641,
        "solve_seed": 273,
        "policy_init_seed": 660,
        "baseline_fn": lambda returns: returns.mean(),
    }

    objective(
        trial=None,
        parameters=parameters,
        total_updates=total_updates,
        env_steps_per_update=64,
        updates_per_eval=100,
        episode_length=64,
        eval_episode_length=64,
        solver_type=solver_type,
        eval_seed=1337,
    )


def search(
    *,
    storage: str = "sqlite:///lqr_pg.sqlite3",
    study_name: str = "policy-gradient-lqr",
    load_if_exists: bool = False,
    n_trials: int = 100,
    total_updates: int = 10_000,
):
    """Perform a hyper parameter search using PolicyGradientSolver on LQR.

    Parameters
    ----------
    storage:
        Storage string. Passed to `optuna.create_study`.
    study_name:
        Name of the study to be used in optuna.
    load_if_exists:
        Continue the study if it already exists.
    n_trials:
        Number of trials to attempt.
    total_updates:
        Number of updates per trial.

    """
    study = optuna.create_study(
        storage=storage,
        study_name=study_name,
        load_if_exists=load_if_exists,
        pruner=optuna.pruners.PercentilePruner(
            percentile=75.0,
            n_warmup_steps=200,
            n_min_trials=3,
        ),
    )

    study.optimize(
        functools.partial(
            objective,
            episode_length=64,
            env_steps_per_update=64,
            total_updates=total_updates,
            updates_per_eval=200,
            eval_episode_length=64,
            suggest_parameters=suggest_parameters,
            eval_seed=1337,
        ),
        n_trials=n_trials,
    )

    # Print best trial info
    best_trial = study.best_trial
    rich.print(f"\nBest trial: {best_trial.number}")
    rich.print(f"Best value: {best_trial.value:.4f}")
    rich.print("Best parameters:")
    for key, value in best_trial.params.items():
        rich.print(f"  {key}: {value}")


if __name__ == "__main__":
    defopt.run([run_simple, search])
