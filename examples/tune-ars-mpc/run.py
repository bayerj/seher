"""Tune MPC with optuna."""

import functools
from typing import Callable

import defopt
import dill
import jax
import jax.random as jr
import optax
import optuna
import plotext

from seher.ars import ars_value_and_grad
from seher.control.mpc import MPCPolicy
from seher.control.stepper_planner import StepperPlanner
from seher.simulate import simulate
from seher.stepper.optax import OptaxOptimizer
from seher.systems.mujoco_playground import MujocoPlaygroundMDP, save_gif
from seher.types import MDP, JaxRandomKey


def _suggest_parameters(trial) -> dict:
    result = dict(
        n_iter=trial.suggest_int("n_iter", low=5, high=10, step=2),
        learning_rate=trial.suggest_float(
            "learning_rate", low=1e-2, high=0.5, log=True
        ),
        std=trial.suggest_float("std", low=0.05, high=1.0, step=0.05),
        n_perturbations=trial.suggest_int(
            "n_perturbations", low=2, high=10, step=2
        ),
        top_k_ratio=trial.suggest_float("top_k_ratio", low=0, high=1),
        n_plan_steps=trial.suggest_int("n_plan_steps", low=4, high=32, step=2),
    )

    print(result)
    return result


def objective(
    trial: optuna.Trial | None,
    mdp: MDP,
    episode_length: int,
    key: JaxRandomKey,
    parameters: dict | None = None,
    suggest_parameters: Callable[..., dict] | None = None,
):
    """Return the performance given the trial."""
    if parameters is None:
        if suggest_parameters is None:
            raise ValueError(
                "either suggest_parameters or parameters must be specified"
            )
        parameters = suggest_parameters(trial)

    top_k = max(
        1, int(parameters["top_k_ratio"] * parameters["n_perturbations"])
    )

    planner = StepperPlanner(
        mdp=mdp,
        n_iter=parameters["n_iter"],
        n_plan_steps=parameters["n_plan_steps"],
        warm_start=True,
        optimizer=OptaxOptimizer(
            objective=None,  # type: ignore
            optimizer=optax.adam(parameters["learning_rate"]),
            value_and_grad=functools.partial(
                ars_value_and_grad,
                std=parameters["std"],
                n_perturbations=parameters["n_perturbations"],
                top_k=top_k,
            ),
        ),
    )
    policy = MPCPolicy(mdp=mdp, planner=planner)  # type: ignore

    batch_simulate = jax.jit(
        jax.vmap(simulate, in_axes=(None, None, None, 0)),
        static_argnums=(0, 1, 2, 7, 8, 9, 10),
    )
    histories = batch_simulate(
        mdp,
        policy,
        episode_length,
        jr.split(key, 8),
    )

    avg_cost = histories.costs.mean()

    if True and isinstance(mdp, MujocoPlaygroundMDP):
        history = jax.tree.map(lambda leaf: leaf[0], histories)
        save_gif(
            mdp,
            history,
            f"rollout-{trial._trial_id if trial is not None else 0}.gif",
        )
    with open("last-histories.dill", "wb") as fp:
        dill.dump(obj=histories, file=fp)

    plotext.clf()
    plotext.plot_size(width=80, height=15)
    for i in range(histories.costs.shape[0]):
        plotext.plot(histories.costs[i].flatten().tolist())
    plotext.title("Cost")

    # Break free from pytest printous -- otherwise the first line of the plot
    # will be started on a pytest printout.
    print()
    plotext.show()

    return avg_cost


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


def search(env: str):
    """Run an optuna study for finding parameters."""
    _, this_objective = _make_mdp_and_objective(env)
    study = optuna.create_study(
        storage="sqlite:///../dmc/db.sqlite3",  # Specify the storage URL here.
        study_name=f"mpc-{env}",
        load_if_exists=True,
    )

    this_objective = functools.partial(
        this_objective, suggest_parameters=_suggest_parameters
    )

    study.optimize(this_objective, n_trials=100)


def run_best(env: str):
    """Run the best parameters for the env."""
    mdp, this_objective = _make_mdp_and_objective(env)
    this_objective(
        trial=None,
        parameters={
            "learning_rate": 0.36996712011558774,
            "n_iter": 9,
            "n_perturbations": 10,
            "n_plan_steps": 10,
            "std": 1.0,
            "top_k_ratio": 0.18113880156136095,
        },
    )


if __name__ == "__main__":
    defopt.run([search, run_best])
