"""Tune MPC with optuna."""

import functools

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import optax
import optuna
import plotext

from seher.apx_arch import MLP, StaticMLPCritic, StaticMLPPolicy
from seher.control.policy_search import (
    ActorCritic,
    ActorCriticCLIProgressCallback,
    ActorCriticOptimizer,
)
from seher.simulate import simulate
from seher.stepper.optax import OptaxOptimizer
from seher.systems.pendulum import SparsePendulumSwingup
from seher.types import MDP, JaxRandomKey


def objective(
    mdp: MDP,
    n_steps: int,
    key: JaxRandomKey,
    trial: optuna.Trial,
    n_time_steps: int,
    n_episode_steps: int,
):
    """Return the performance given the trial."""
    # Set Parameters.

    discount = trial.suggest_float("discount", low=0.9, high=0.99, step=0.01)

    total_steps = 20_000
    learning_rate = trial.suggest_float(
        "learning_rate", low=1e-3, high=0.1, step=1e-3
    )
    n_simulations = trial.suggest_int("n_simulations", low=4, high=32)
    td_lambda = trial.suggest_float(
        "td_lambda", low=0.9, high=0.999, step=1e-3
    )
    critic_weight = trial.suggest_float(
        "critic_weight", low=0.05, high=10.0, step=0.05
    )
    max_norm = trial.suggest_categorical("max_norm", [0.1, 1.0, 10.0, 100.0])
    polyak_step_size = trial.suggest_float(
        "polyak_step_size", low=0.001, high=0.5, log=True
    )

    policy_use_layernorm = trial.suggest_categorical(
        "policy_use_layernorm", [True, False]
    )
    critic_use_layernorm = trial.suggest_categorical(
        "critic_use_layernorm", [True, False]
    )

    critic_init_seed = trial.suggest_int("critic_init_seed", low=1, high=32)
    policy_init_seed = trial.suggest_int("policy_init_seed", low=1, high=32)

    # Setup objects.

    mdp = mdp.replace(discount=discount)  # type: ignore

    actor_critic = ActorCriticOptimizer(
        mdp=mdp,
        n_simulations=n_simulations,
        n_steps=n_time_steps,
        td_lambda=td_lambda,
        critic_weight=critic_weight,
        optimizer=OptaxOptimizer(
            objective=None,  # type: ignore
            optimizer=optax.chain(
                optax.clip_by_global_norm(max_norm),
                optax.adam(learning_rate),
            ),
        ),
        steps_per_init=n_episode_steps,
        polyak_step_size=polyak_step_size,
    )

    policy = StaticMLPPolicy(
        # obs_to_array=lambda state: state,  # type: ignore
        obs_to_array=lambda state: state.cos_sin_repr(),  # type: ignore
        array_to_control=lambda x: x,
        mlp=MLP.make(
            inpt_size=3,
            layer_sizes=[32],
            output_size=1,
            activations=[
                jnn.soft_sign,
                lambda x: jnn.soft_sign(x) * 4 - 2,
            ],
            key=jr.PRNGKey(policy_init_seed),
        ).replace(use_layernorm=policy_use_layernorm),  # type: ignore
    )

    critic = StaticMLPCritic(
        # state_to_array=lambda state: state,  # type: ignore
        state_to_array=lambda state: state.cos_sin_repr(),  # type: ignore
        mlp=MLP.make(
            inpt_size=3,
            layer_sizes=[32],
            output_size=1,
            activations=[
                jnn.swish,
                lambda x: -jnn.swish(x),
            ],
            key=jr.PRNGKey(critic_init_seed),
        ).replace(use_layernorm=critic_use_layernorm),  # type: ignore
    )

    # Prepare training.

    carry = actor_critic.initial_carry(
        sample_parameter=ActorCritic(
            actor=policy,
            critic=critic,
            target_critic=jt.map(lambda x: x, critic),
        )
    )

    call_actor_critic = jax.jit(actor_critic.__call__)
    callback = ActorCriticCLIProgressCallback(total_steps=total_steps)
    batch_simulate = jax.vmap(simulate, in_axes=(None, None, None, 0))
    keys = jr.split(key, 8)

    # Train.

    for i in range(total_steps):
        key, search_key = jr.split(key, 2)
        carry, solution, aux = call_actor_critic(carry, None, search_key)
        callback(i, aux)

    callback.teardown()

    history = batch_simulate(actor_critic.mdp, solution.actor, n_steps, keys)

    plotext.plot_size(height=30, width=80)
    for i in range(history.costs.shape[0]):
        plotext.plot(history.costs[i].flatten().tolist())
    plotext.show()

    return jnp.array(history.costs[:, -10:].mean())


mdp = SparsePendulumSwingup()
this_objective = functools.partial(
    objective, mdp, 100, jr.PRNGKey(1), n_episode_steps=32, n_time_steps=16
)

study = optuna.create_study(
    storage="sqlite:///../dmc/db.sqlite3",  # Specify the storage URL here.
    study_name="tune-ac-sparse-pend-swingup",
    load_if_exists=True,
)

study.optimize(this_objective, n_trials=100)  # type: ignore
