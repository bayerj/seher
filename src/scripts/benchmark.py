"""Script to benchmark certain key-computations."""

import contextlib
import functools
import time

import jax
import jax.nn as jnn
import jax.random as jr
import optax
import rich

from seher.apx_arch import MLP, StaticMLPPolicy
from seher.ars import ars_grad
from seher.control.mpc import calc_costs_of_plan
from seher.control.policy_search import MCPS
from seher.stepper.optax import OptaxOptimizer
from seher.systems.mujoco_playground import MujocoPlaygroundMDP


@contextlib.contextmanager
def _measure(pars):
    start = time.time()
    yield
    stop = time.time()
    rich.print({**pars, "duration": stop - start})


def _benchmark_calc_costs_of_plan(
    mdp_handle, n_simulations, n_steps, n_repeats
):
    pars = locals()
    mdp = MujocoPlaygroundMDP.from_registry(mdp_handle)

    plans = jr.uniform(
        shape=(n_simulations, n_steps, mdp.empty_control().shape[0]),
        key=jr.PRNGKey(23),
    )
    keys = jr.split(jr.PRNGKey(32), n_simulations)

    initial_state = mdp.init(jr.PRNGKey(1))
    # initial_states = einops.repeat(
    # initial_state, "d -> n d", n=n_simulations
    # )

    compute = jax.vmap(calc_costs_of_plan, in_axes=(None, 0, None, 0))

    for i in range(n_repeats):
        with _measure({"id": i, **pars}):
            compute(mdp, plans, initial_state, keys).block_until_ready()


def _benchmark_ars_policy_search_step(
    mdp_handle,
    n_simulations,
    n_steps,
    n_repeats,
    policy_layers,
    n_candidates,
    top_k,
):
    pars = locals()

    def obs_to_array(state):
        return state.obs

    policy_search = MCPS(
        mdp=MujocoPlaygroundMDP.from_registry(mdp_handle).replace(  # type: ignore
            n_inner_steps=2
        ),
        n_simulations=n_simulations,
        n_steps=n_steps,
        optimizer=OptaxOptimizer(
            objective=None,  # type: ignore
            optimizer=optax.adam(3e-3),
            has_aux=False,
            grad=functools.partial(
                ars_grad,
                std=0.2,
                n_perturbations=n_candidates,
                top_k=top_k,
            ),
        ),
    )

    key = jr.PRNGKey(12)
    policy_init_key, key = jr.split(key)

    obs_dim = obs_to_array(policy_search.mdp.init(key=key)).shape[0]
    n_simulations = 8

    activations = [jnn.soft_sign] * len(policy_layers) + [
        lambda x: jnn.soft_sign(x) * 4 - 2
    ]

    policy = StaticMLPPolicy(
        obs_to_array=obs_to_array,  # type: ignore
        array_to_control=lambda x: x,
        mlp=MLP.make(
            inpt_size=obs_dim,
            layer_sizes=policy_layers,
            output_size=policy_search.mdp.empty_control().shape[0],  # type: ignore
            activations=activations,
            key=policy_init_key,
        ),
    )
    carry = policy_search.initial_carry(sample_parameter=policy)

    call_policy_search = jax.jit(policy_search.__call__)

    for i in range(n_repeats):
        with _measure({"id": i, **pars}):
            _, _, _ = call_policy_search(carry, None, key)
            # history.costs.block_until_ready()


if __name__ == "__main__":
    # _benchmark_calc_costs_of_plan("CartpoleBalance", 512, 1000, 2)
    # _benchmark_calc_costs_of_plan("CartpoleBalance", 512, 100, 2)
    # _benchmark_calc_costs_of_plan("HumanoidStand", 2, 30, 2)
    # _benchmark_calc_costs_of_plan("HumanoidStand", 2, 10, 2)

    # _benchmark_ars_policy_search_step(
    # "CartpoleBalance",
    # n_simulations=2,
    # n_steps=4,
    # n_repeats=5,
    # policy_layers=[8],
    # n_candidates=2,
    # top_k=1,
    # jit_policy=True,
    # )

    _benchmark_ars_policy_search_step(
        "CartpoleBalance",
        n_simulations=4,
        n_steps=10,
        n_repeats=5,
        policy_layers=[512, 256, 128],
        n_candidates=4,
        top_k=1,
    )
