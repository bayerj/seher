"""Test integrating MPC with MDPs."""

import functools

import jax.numpy as jnp
import jax.random as jr
import optax
import plotext
import pytest

from seher.ars import ars_value_and_grad
from seher.control.mpc import MPCPolicy, RandomSearchPlanner
from seher.control.stepper_planner import StepperPlanner
from seher.simulate import RichProgressCallback, simulate
from seher.stepper.mppi import GaussianMPPIOptimizer
from seher.stepper.optax import OptaxOptimizer
from seher.systems.mujoco_playground import MujocoPlaygroundMDP, save_gif
from seher.systems.pendulum import Pendulum
from seher.types import JaxRandomKey


def _uniform_proposal(
    n_candidates: int,
    n_steps: int,
    control_dim: int,
    key: JaxRandomKey,
    min_control: float,
    max_control: float,
):
    return jr.uniform(
        key=key,
        shape=(n_candidates, n_steps, control_dim),
        minval=min_control,
        maxval=max_control,
    )


@pytest.mark.parametrize(
    "handle,n_steps,planner,mdp,key,target_cost",
    [
        (
            "random-search-on-pendulum",
            100,
            RandomSearchPlanner(
                mdp=None,  # type: ignore
                n_candidates=20,
                n_plan_steps=10,
                proposal=functools.partial(
                    _uniform_proposal, min_control=-2, max_control=2
                ),
            ),
            Pendulum(),
            jr.PRNGKey(42),
            4.0,
        ),
        (
            "mppi-search-on-pendulum",
            100,
            StepperPlanner(
                mdp=None,  # type: ignore
                n_iter=5,
                n_plan_steps=10,
                warm_start=True,
                optimizer=GaussianMPPIOptimizer(
                    objective=None,
                    n_candidates=16,
                    top_k=4,
                    min_scale=0.01,
                    temperature=1.0,
                    initial_loc=jnp.array(0.0),
                    initial_scale=jnp.array(3.0),
                ),
            ),
            Pendulum(),
            jr.PRNGKey(42),
            3.0,
        ),
        (
            "optax-analytics-gradient-on-pendulum",
            100,
            StepperPlanner(
                mdp=None,  # type: ignore
                n_iter=50,
                n_plan_steps=20,
                warm_start=True,
                optimizer=OptaxOptimizer(
                    objective=None,  # type: ignore
                    optimizer=optax.adam(1e-1),
                    value_and_grad=functools.partial(
                        ars_value_and_grad,
                        std=0.1,
                        n_perturbations=16,
                        top_k=4,
                    ),
                ),
            ),
            Pendulum(),
            jr.PRNGKey(42),
            3.0,
        ),
        pytest.param(
            "mppi-on-cartpole",
            150,
            StepperPlanner(
                mdp=None,  # type: ignore
                n_iter=5,
                n_plan_steps=20,
                warm_start=True,
                optimizer=GaussianMPPIOptimizer(
                    objective=None,
                    n_candidates=32,
                    top_k=4,
                    min_scale=0.01,
                    temperature=0.5,
                    initial_loc=jnp.array(0.0),
                    initial_scale=jnp.array(0.5),
                ),
            ),
            MujocoPlaygroundMDP.from_registry("CartpoleSwingup").replace(  # type: ignore
                n_inner_steps=4
            ),
            jr.PRNGKey(47),
            -0.62,
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "ars-mpc-on-cartpole",
            75,
            StepperPlanner(
                mdp=None,  # type: ignore
                n_iter=5,
                n_plan_steps=6,
                warm_start=True,
                optimizer=OptaxOptimizer(
                    objective=None,  # type: ignore
                    optimizer=optax.adam(0.046),
                    value_and_grad=functools.partial(
                        ars_value_and_grad,
                        std=1.0,
                        n_perturbations=2,
                        top_k=1,
                    ),
                ),
            ),
            MujocoPlaygroundMDP.from_registry("CartpoleBalance").replace(  # type: ignore
                n_inner_steps=4
            ),
            jr.PRNGKey(2),
            -0.97,
            marks=pytest.mark.slow,
        ),
    ],
)
def test_mpc_on_mdp(handle, mdp, n_steps, planner, key, target_cost):
    """Test whether various MPC policies solve various environments."""
    planner = planner.replace(mdp=mdp)
    policy = MPCPolicy(mdp=mdp, planner=planner)

    history = simulate(
        policy=policy,
        mdp=mdp,
        key=key,
        n_steps=n_steps,
        callback=RichProgressCallback(total_steps=n_steps),
    )
    avg_cost = history.costs.mean()
    print(f"{avg_cost=:.4f}")

    plotext.clf()
    plotext.plot_size(width=80, height=15)
    plotext.plot(history.costs.flatten().tolist())
    plotext.title("Cost")

    # Break free from pytest printous -- otherwise the first line of the plot
    # will be started on a pytest printout.
    print()
    plotext.show()

    if isinstance(mdp, MujocoPlaygroundMDP):
        save_gif(mdp, history, f"test-artifacts/{handle}.gif")

    assert avg_cost <= target_cost, "performance not good enough"
