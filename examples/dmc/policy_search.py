"""Tune policy search with optuna."""

import dataclasses
import functools
import pathlib
from typing import Callable

import defopt
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import optax
import optuna
import optuna.pruners
import rich
from mujoco_playground import registry as mp_registry
from parameters import parameters_policy_search as parameters

from seher.apx_arch import symlog
from seher.ars import ars_grad
from seher.control.policy_search import (
    MCPSCallback,
    MCPSCLIProgressCallback,
    SaveRolloutGif,
    WriteEvalCostsToFile,
)
from seher.control.solvers import PolicySearchSolver, plot_simulation_costs
from seher.systems.mujoco_playground import MujocoPlaygroundMDP, save_gif
from seher.types import JaxRandomKey


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


def suggest_parameters(trial) -> dict:
    """Return solver-compatible parameters for optuna trial."""
    ars_n_perturbations = trial.suggest_int(
        "ars_n_perturbations", low=2, high=8
    )
    n_hidden = trial.suggest_int("policy.n_hidden", low=16, high=128)
    activation_name = trial.suggest_categorical(
        "activation", ["soft_sign", "swish", "relu"]
    )

    # Convert activation string to actual function
    activation_fn = getattr(jnn, activation_name)

    return {
        # MDP configuration
        "mdp_kws": {
            "discount": trial.suggest_float(
                "discount", low=0.9, high=1.0, step=0.001
            )
        },
        # Core solver parameters
        "n_simulations": trial.suggest_int("n_simulations", low=1, high=8),
        "seed": trial.suggest_int("seed", low=0, high=1337),
        # Optimizer configuration
        "optimizer_kws": {
            "optimizer": optax.chain(
                optax.clip_by_global_norm(
                    trial.suggest_categorical(
                        "grad_max_norm", [0.1, 1.0, 10.0, 100.0]
                    )
                ),
                getattr(
                    optax,
                    trial.suggest_categorical("optimizer", ["adam", "sgd"]),
                )(
                    trial.suggest_float(
                        "learning_rate", low=0.0001, high=0.3, step=0.0001
                    )
                ),
            ),
            "grad": functools.partial(
                ars_grad,
                std=trial.suggest_float(
                    "ars_std", low=0.0001, high=0.5, step=0.0001
                ),
                n_perturbations=ars_n_perturbations,
                top_k=trial.suggest_int(
                    "ars_top_k", low=1, high=ars_n_perturbations
                ),
            ),
            "has_aux": True,
        },
        # Policy MLP configuration
        "policy_mlp_kws": {
            "layer_sizes": [n_hidden],
            "activations": [activation_fn, lambda x: jnn.soft_sign(x) * 2 - 1],
            "use_layernorm": trial.suggest_categorical(
                "use_layernorm", [False, True]
            ),
        },
    }


def objective(
    trial: optuna.Trial | None,
    mdp: MujocoPlaygroundMDP,
    key: JaxRandomKey,
    episode_length: int,
    env_steps_per_update: int,
    updates_per_eval: int,
    total_updates: int = 20_000,
    save_gifs: bool = True,
    parameters: dict | None = None,
    suggest_parameters: Callable[[optuna.Trial], dict] = suggest_parameters,
    eval_episode_length: int | None = None,
):
    """Return the performance given the trial or parameters."""
    if parameters is None:
        if not isinstance(trial, optuna.Trial):
            raise ValueError("either parameters or trial must not be None")
        parameters = suggest_parameters(trial)

    if eval_episode_length is None:
        eval_episode_length = episode_length

    # Create solver with parameters
    mdp_kws = parameters.get("mdp_kws", {})
    seed = parameters.get("seed", 0)
    solver_params = {
        k: v for k, v in parameters.items() if k not in ["mdp_kws", "seed"]
    }
    solver = PolicySearchSolver(
        obs_to_array=lambda state: symlog(state.obs),
        episode_length=episode_length,
        steps_per_update=env_steps_per_update,
        max_updates=total_updates,
        updates_per_eval=updates_per_eval,
        eval_n_simulations=16,
        **solver_params,
    )

    if trial:
        solver.callbacks.append(OptunaTrialCallback(trial=trial))

    callbacks = _make_callbacks(
        mdp,
        save_gifs=save_gifs,
        write_costs_to_file=False,  # Make available through CLI.
        show_progress=True,  # Make available through CLI.
        total_updates=total_updates,
    )
    solver.callbacks += callbacks

    key = jr.PRNGKey(seed)
    mdp = mdp.replace(**mdp_kws)  # type: ignore
    solver.solve(mdp, key=key)
    history = solver.simulate(n_steps=eval_episode_length, n_simulations=16)

    if save_gifs and isinstance(mdp, MujocoPlaygroundMDP):
        one_history = jax.tree.map(lambda leaf: leaf[0], history)
        trial_id = 0 if trial is None else trial._trial_id
        save_gif(
            mdp, one_history, f"rollout-{trial_id}-{solver.max_updates}.gif"
        )

    plot_simulation_costs(history=history)

    final_avg_cost = jnp.array(history.costs.mean())
    rich.print(f"Final average cost: {final_avg_cost:.4f}")
    return final_avg_cost


def _make_mdp_and_objective(env: str) -> tuple[MujocoPlaygroundMDP, Callable]:
    mdp = MujocoPlaygroundMDP.from_registry(env).replace(  # type: ignore
        n_inner_steps=2
    )
    this_objective = functools.partial(
        objective,
        mdp=mdp,
        key=jr.PRNGKey(1),
        episode_length=500,
    )

    return mdp, this_objective


def _make_callbacks(
    mdp: MujocoPlaygroundMDP,
    save_gifs: bool,
    show_progress: bool,
    write_costs_to_file: bool,
    total_updates: int,
) -> list[MCPSCallback]:
    callbacks = []

    if save_gifs:
        callbacks.append(SaveRolloutGif(mdp))

    if show_progress:
        callbacks.append(
            MCPSCLIProgressCallback(
                total_steps=total_updates,
            )
        )

    if write_costs_to_file:
        callbacks.append(
            WriteEvalCostsToFile(file_name=pathlib.Path("costs.txt"))
        )

    return callbacks


def run_best(env: str, save_gifs: bool = False, total_updates: int = 20_000):
    """Run the best parameters for the envs.

    Parameters
    ----------
    env:
        Environment to use.
    save_gifs:
        If True, save gifs of the rollouts.
    total_updates:
        Number of updates to perform.

    """
    _, this_objective = _make_mdp_and_objective(env)

    if env not in parameters:
        raise ValueError(f"no best parameters found for: {env}")

    this_objective(
        trial=None,
        parameters=parameters[env],
        total_updates=total_updates,
        env_steps_per_update=50,
        updates_per_eval=50,
        episode_length=500,
        eval_episode_length=500,
        save_gifs=save_gifs,
    )


def search(
    env: str,
    *,
    storage: str = "sqlite:///db.sqlite3",
    study_name: str | None = None,
    load_if_exists: bool = False,
    n_trials: int = 1,
    save_gifs: bool = False,
):
    """Perform a hyper parameter search on the environment.

    A standard search space is used.

    Parameters
    ----------
    env:
        Environment to run HPS on. See `mujoco_playground.registry.ALL_ENVS`
        for a list.
    storage:
        Storage string. Passed to `optuna.create_study`.
    study_name:
        Name of the study to be used in optuna. If it already exists, runs will
        be added to this study only if `load_if_exists` is True.
    load_if_exists:
        Continue the study if it already exists.
    n_trials:
        Number of trials to attempt.
    save_gifs:
        If True, save gifs of the rollouts.

    """
    total_updates = 50_000
    _, this_objective = _make_mdp_and_objective(env)

    if study_name is None:
        study_name = f"policy-search-{env}"

    study = optuna.create_study(
        storage=storage,  # Specify the storage URL here.
        study_name=study_name,
        load_if_exists=load_if_exists,
        pruner=optuna.pruners.PercentilePruner(
            percentile=75.0,
            n_warmup_steps=500,
            n_min_trials=3,
        ),
    )

    study.optimize(
        functools.partial(  # type: ignore
            this_objective,
            save_gifs=save_gifs,
            env_steps_per_update=50,
            total_updates=total_updates,
            updates_per_eval=1000,
            episode_length=500,
            suggest_parameters=suggest_parameters,
            eval_episode_length=500,
        ),
        n_trials=n_trials,
    )  # type: ignore


def list_envs():
    """List all environments of mujoco playground."""
    print(", ".join(mp_registry.ALL_ENVS))


if __name__ == "__main__":
    defopt.run([run_best, search, list_envs])
