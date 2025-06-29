"""Parameter values that have been shown to work well."""

import functools

import jax.nn as jnn
import optax

from seher.ars import ars_grad

parameters_policy_search = {
    "WalkerWalk": {
        # MDP configuration
        "mdp_kws": {"discount": 1.0},
        # Core solver parameters
        "n_simulations": 8,
        "seed": 655,
        # Optimizer configuration
        "optimizer_kws": {
            "optimizer": optax.chain(
                optax.clip_by_global_norm(10.0),
                optax.sgd(0.2033),
            ),
            "grad": functools.partial(
                ars_grad,
                std=0.41500000000000004,
                n_perturbations=8,
                top_k=8,
            ),
            "has_aux": True,
        },
        # Policy MLP configuration
        "policy_mlp_kws": {
            "layer_sizes": [114],
            "activations": [jnn.relu, lambda x: jnn.soft_sign(x) * 2 - 1],
            "use_layernorm": True,
        },
    },
    "CartpoleBalance": {
        # MDP configuration
        "mdp_kws": {"discount": 0.9520000000000001},
        # Core solver parameters
        "n_simulations": 2,
        "seed": 149,
        # Optimizer configuration
        "optimizer_kws": {
            "optimizer": optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(0.014),
            ),
            "grad": functools.partial(
                ars_grad,
                std=0.3425,
                n_perturbations=2,
                top_k=1,
            ),
            "has_aux": True,
        },
        # Policy MLP configuration
        "policy_mlp_kws": {
            "layer_sizes": [74],
            "activations": [jnn.relu, lambda x: jnn.soft_sign(x) * 2 - 1],
            "use_layernorm": False,
        },
    },
}
