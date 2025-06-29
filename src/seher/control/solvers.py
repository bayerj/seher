"""High-level solver API for reinforcement learning on MDPs and POMDPs."""

import abc
import dataclasses
import functools
from typing import Callable, TypeVar, Union, cast

import jax
import jax.nn as jnn
import jax.random as jr
import optax

from seher.apx_arch import MLP, StaticMLPCritic, StaticMLPPolicy
from seher.control.policy_search import (
    MCPS,
    ActorCritic,
    ActorCriticEnsembleOptimizer,
    ActorCriticOptimizer,
    PolicyGradientOptimizer,
)
from seher.control.tanh_gaussian_policy_mdp import (
    SimulationPolicy,
    TanhGaussianPolicyControl,
)
from seher.jax_util import tree_stack
from seher.simulate import History, simulate
from seher.stepper.optax import OptaxOptimizer
from seher.types import (
    CMDP,
    CPOMDP,
    MDP,
    POMDP,
    JaxRandomKey,
    Policy,
    StateCritic,
)

from .tanh_gaussian_policy_mdp import (
    convert_tanh_gaussian_history_to_original,
)

# Generic for different problem types
ProblemType = TypeVar("ProblemType", bound=Union[MDP, POMDP, CMDP, CPOMDP])


def _create_mlp_policy(
    obs_to_array: Callable,
    obs_dim: int,
    control_dim: int,
    key: JaxRandomKey,
    **mlp_kws,
) -> StaticMLPPolicy:
    """Create a StaticMLPPolicy with MLP architecture."""
    defaults = {
        "layer_sizes": [32],
        "activations": [jnn.soft_sign, lambda x: jnn.soft_sign(x) * 4 - 2],
    }
    defaults.update(mlp_kws)

    return StaticMLPPolicy(
        obs_to_array=obs_to_array,
        array_to_control=lambda x: x,
        mlp=MLP.make(
            inpt_size=obs_dim, output_size=control_dim, key=key, **defaults
        ),
    )


def _create_mlp_critic(
    state_to_array: Callable, state_dim: int, key: JaxRandomKey, **mlp_kws
) -> StaticMLPCritic:
    """Create a StaticMLPCritic with MLP architecture."""
    defaults = {
        "layer_sizes": [32],
        "activations": [jnn.soft_sign, lambda x: 20 * jnn.soft_sign(x) - 10],
    }
    defaults.update(mlp_kws)

    return StaticMLPCritic(
        state_to_array=state_to_array,
        mlp=MLP.make(inpt_size=state_dim, output_size=1, key=key, **defaults),
    )


def _create_optimizer(
    optax_optimizer: str,
    optax_optimizer_kws: dict,
    optimizer_kws: dict,
    max_grad_norm: float | None = None,
) -> OptaxOptimizer:
    """Create an OptaxOptimizer with dynamic optax optimizer."""
    if "optimizer" not in optimizer_kws:
        optax_fn = getattr(optax, optax_optimizer)
        optax_opt = optax_fn(**optax_optimizer_kws)

        if max_grad_norm is not None:
            optax_opt = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax_opt,
            )

        optimizer_kws = {"optimizer": optax_opt, **optimizer_kws}

    result = OptaxOptimizer(objective=None, **optimizer_kws)  # type: ignore

    return result


@dataclasses.dataclass
class BaseSolver[ProblemType](abc.ABC):
    """Base class for RL solvers with common functionality."""

    callbacks: list = dataclasses.field(default_factory=list)
    episode_length: int = 64
    steps_per_update: int = 64
    updates_per_eval: int | None = None
    eval_n_simulations: int = 16
    n_simulations: int = 32
    max_updates: int = 25_000
    max_env_steps: int | None = None

    # Public attributes
    # TODO make problem non-writable, or undo jitting on write.
    problem: ProblemType | None = dataclasses.field(init=False, default=None)
    policy: Policy | None = dataclasses.field(init=False, default=None)
    history: History | None = dataclasses.field(init=False, default=None)
    is_solved: bool = dataclasses.field(init=False, default=False)
    prepared: bool = dataclasses.field(init=False, default=False)
    _jit_batch_simulate: Callable | None = dataclasses.field(
        init=False, default=None
    )

    def __post_init__(self):
        """Validate episode_length and steps_per_update relationship."""
        if self.episode_length % self.steps_per_update != 0:
            raise ValueError(
                f"episode_length ({self.episode_length}) must be divisible by "
                f"steps_per_update ({self.steps_per_update})"
            )

    @abc.abstractmethod
    def solve(
        self, problem: ProblemType, key: JaxRandomKey | None = None, **kwargs
    ):
        """Solve the problem."""
        pass

    def _run_training_loop(
        self,
        optimizer_instance,
        initial_carry,
        train_step_fn: Callable,
        key: JaxRandomKey,
        eval_key: JaxRandomKey,
    ):
        """Run generic training loop."""
        del optimizer_instance, eval_key
        total_env_steps = 0

        carry = initial_carry
        for i in range(self.max_updates):
            key, search_key = jr.split(key, 2)
            carry, solution, aux = train_step_fn(carry, None, search_key)

            # Let subclass update its specific attributes
            self._update_solution_attributes(solution)

            total_env_steps += self.n_simulations * self.steps_per_update

            eval_history = None
            if (
                self.updates_per_eval is not None
                and i % self.updates_per_eval == 0
            ):
                eval_history = self.simulate(
                    n_steps=self.episode_length,
                    n_simulations=self.eval_n_simulations,
                )

            for callback in self.callbacks:
                callback(
                    i,
                    train_history=aux.history,
                    eval_history=eval_history,
                    aux=aux,
                )

            if (
                self.max_env_steps is not None
                and total_env_steps >= self.max_env_steps
            ):
                break

        for callback in self.callbacks:
            if hasattr(callback, "teardown"):
                callback.teardown()

        return solution, aux

    @abc.abstractmethod
    def _update_solution_attributes(self, solution):
        """Update solver-specific attributes with the solution."""
        pass

    def simulate(
        self,
        n_steps: int = 100,
        n_simulations: int = 1,
        key: JaxRandomKey | None = None,
    ) -> History:
        """Simulate the learned policy."""
        if not self.prepared:
            raise ValueError(
                "Solver must be prepared before simulation. "
                "Call .solve() first."
            )

        if key is None:
            key = jr.PRNGKey(0)

        if self._jit_batch_simulate is None:
            # XXX When we move beyond MDPs, we will have to adapt simulate.
            problem = cast(MDP, self.problem)

            this_simulate = functools.partial(simulate, problem)
            self._jit_batch_simulate = jax.jit(
                jax.vmap(this_simulate, in_axes=(None, None, 0)),
                static_argnums=(1, 6, 7, 8, 9),
            )

        keys = jr.split(key, n_simulations)
        return self._jit_batch_simulate(self.policy, n_steps, keys)

    def score(
        self,
        n_simulations: int,
        n_steps: int,
        key: JaxRandomKey,
    ) -> float:
        """Evaluate current policy performance."""
        if not self.prepared:
            raise ValueError(
                "Solver must be prepared before scoring. Call .solve() first."
            )

        keys = jr.split(key, n_simulations)
        jit_batch_simulate = jax.jit(
            jax.vmap(simulate, in_axes=(None, None, None, 0)),
            static_argnums=(0, 2, 7, 8, 9, 10),
        )
        history = jit_batch_simulate(self.problem, self.policy, n_steps, keys)

        # XXX Undo this cast when we move beyond MDPs.
        problem = cast(MDP, self.problem)

        discount_factors = problem.discount ** jax.numpy.arange(n_steps)
        discounted_costs = (discount_factors * history.costs).sum(axis=1)
        return float(discounted_costs.mean())


@dataclasses.dataclass
class PolicySearchSolver[State, Control, Cost](
    BaseSolver[MDP[State, Control, Cost]]
):
    """Scikit-learn style solver for policy search on MDPs/POMDPs."""

    obs_to_array: Callable = lambda obs: obs
    n_simulations: int = 32
    max_updates: int = 25_000
    max_env_steps: int | None = None

    # Optimizer configuration
    optax_optimizer: str = "adam"
    optax_optimizer_kws: dict = dataclasses.field(
        default_factory=lambda: {"learning_rate": 0.01}
    )
    optimizer_kws: dict = dataclasses.field(default_factory=dict)

    # Policy configuration
    policy_mlp_kws: dict = dataclasses.field(
        default_factory=lambda: {"layer_sizes": [32]}
    )

    # Public attributes
    costs: jax.Array | None = dataclasses.field(init=False, default=None)

    def _update_solution_attributes(self, solution):
        """Update policy with the solution."""
        self.policy = solution

    def _prepare(
        self,
        problem: MDP,
        policy_init_key: JaxRandomKey,
    ) -> StaticMLPPolicy[State, Control]:
        """Initialize policy network."""
        sample_state = problem.init(key=policy_init_key)
        control_dim = problem.empty_control().shape[0]
        obs_dim = self.obs_to_array(sample_state).shape[0]

        policy = _create_mlp_policy(
            obs_to_array=self.obs_to_array,
            obs_dim=obs_dim,
            control_dim=control_dim,
            key=policy_init_key,
            **self.policy_mlp_kws,
        )

        self.policy = policy
        self.prepared = True
        return policy

    def solve(
        self,
        problem: MDP,
        key: JaxRandomKey,
        eval_key: JaxRandomKey | None = None,
        policy_init_key: JaxRandomKey | None = None,
    ):
        """Solve the MDP using policy search."""
        self.problem = problem

        if eval_key is None:
            eval_key, key = jr.split(key)
            eval_key = cast(JaxRandomKey, eval_key)

        if policy_init_key is None:
            policy_init_key, key = jr.split(key)
            policy_init_key = cast(JaxRandomKey, policy_init_key)

        if not self.prepared:
            policy = self._prepare(problem, policy_init_key)
        else:
            policy = self.policy

        if not isinstance(policy, StaticMLPPolicy):
            raise ValueError("policy must be of class `StaticMLPPolicy`")

        optimizer = _create_optimizer(
            self.optax_optimizer, self.optax_optimizer_kws, self.optimizer_kws
        )

        policy_search = MCPS(
            mdp=problem,
            policy=policy,
            n_simulations=self.n_simulations,
            n_steps=self.steps_per_update,
            # The type below is not fixed yet here, since we will set the
            # objective to None. This is a flaw in our type system, which
            # needs to be fixed eventually.
            optimizer=optimizer,  # type: ignore
            steps_per_init=self.episode_length,
        )

        carry = policy_search.initial_carry(sample_parameter=policy)
        call_policy_search = jax.jit(policy_search.__call__)

        solution, aux = self._run_training_loop(
            policy_search,
            carry,
            call_policy_search,
            key=key,
            eval_key=eval_key,
        )

        self.policy = solution
        self.history = aux.history
        self.is_solved = True


@dataclasses.dataclass
class ActorCriticSolver(BaseSolver[MDP]):
    """Scikit-learn style solver for actor-critic on MDPs/POMDPs."""

    obs_to_array: Callable = lambda obs: obs
    state_to_array: Callable = lambda state: state
    episode_length: int = 32
    steps_per_update: int = 32
    n_simulations: int = 32
    max_updates: int = 25_000
    max_env_steps: int | None = None
    td_lambda: float = 0.95
    critic_weight: float = 1.0
    polyak_step_size: float | None = None

    # Optimizer configuration
    optax_optimizer: str = "adam"
    optax_optimizer_kws: dict = dataclasses.field(
        default_factory=lambda: {"learning_rate": 0.01}
    )
    optimizer_kws: dict = dataclasses.field(default_factory=dict)

    # Network configuration
    policy_mlp_kws: dict = dataclasses.field(
        default_factory=lambda: {"layer_sizes": [32]}
    )
    critic_mlp_kws: dict = dataclasses.field(
        default_factory=lambda: {"layer_sizes": [32]}
    )

    # Public attributes
    actor: Policy | None = dataclasses.field(init=False, default=None)
    critic: StateCritic | None = dataclasses.field(init=False, default=None)

    def _update_solution_attributes(self, solution):
        """Update actor, critic and policy with the solution."""
        self.actor = solution.actor
        self.critic = solution.critic
        self.policy = solution.actor

    def _prepare(
        self,
        problem: MDP,
        policy_init_key: JaxRandomKey,
        critic_init_key: JaxRandomKey,
    ):
        """Initialize actor and critic networks."""
        sample_state = problem.init(key=policy_init_key)
        control_dim = problem.empty_control().shape[0]
        obs_dim = self.obs_to_array(sample_state).shape[0]
        state_dim = self.state_to_array(sample_state).shape[0]

        actor = _create_mlp_policy(
            obs_to_array=self.obs_to_array,
            obs_dim=obs_dim,
            control_dim=control_dim,
            key=policy_init_key,
            **self.policy_mlp_kws,
        )

        critic = _create_mlp_critic(
            state_to_array=self.state_to_array,
            state_dim=state_dim,
            key=critic_init_key,
            **self.critic_mlp_kws,
        )

        self.actor = actor
        self.critic = critic
        self.policy = actor  # For base class compatibility
        self.prepared = True
        return actor, critic

    def solve(
        self,
        problem: MDP,
        key: JaxRandomKey,
        eval_key: JaxRandomKey | None = None,
        policy_init_key: JaxRandomKey | None = None,
        critic_init_key: JaxRandomKey | None = None,
    ):
        """Solve the MDP using actor-critic."""
        self.problem = problem

        if eval_key is None:
            eval_key, key = jr.split(key)
            eval_key = cast(JaxRandomKey, eval_key)

        if policy_init_key is None:
            policy_init_key, key = jr.split(key)
            policy_init_key = cast(JaxRandomKey, policy_init_key)

        if critic_init_key is None:
            critic_init_key, key = jr.split(key)
            critic_init_key = cast(JaxRandomKey, critic_init_key)

        # Only prepare networks if not already prepared
        if not self.prepared:
            actor, critic = self._prepare(
                problem, policy_init_key, critic_init_key
            )
        else:
            actor = self.actor
            critic = self.critic

        if not isinstance(actor, StaticMLPPolicy):
            raise ValueError("actor needs to be of type StaticMLPPolicy")
        if not isinstance(critic, StaticMLPCritic):
            raise ValueError("actor needs to be of type StaticMLPCritic")

        target_critic = (
            jax.tree.map(lambda x: x, critic)
            if self.polyak_step_size is not None
            else None
        )

        optimizer = _create_optimizer(
            self.optax_optimizer, self.optax_optimizer_kws, self.optimizer_kws
        )

        actor_critic_optimizer = ActorCriticOptimizer(
            mdp=problem,
            n_simulations=self.n_simulations,
            n_steps=self.steps_per_update,
            # The type below is not fixed yet here, since we will set the
            # objective to None. This is a flaw in our type system, which
            # needs to be fixed eventually.
            optimizer=optimizer,  # type: ignore
            critic_weight=self.critic_weight,
            td_lambda=self.td_lambda,
            steps_per_init=self.episode_length,
            polyak_step_size=self.polyak_step_size,
        )

        actor_critic = ActorCritic(
            actor=actor,
            critic=critic,
            target_critic=target_critic,
        )
        carry = actor_critic_optimizer.initial_carry(
            sample_parameter=actor_critic
        )
        call_actor_critic = jax.jit(actor_critic_optimizer.__call__)

        solution, aux = self._run_training_loop(
            actor_critic_optimizer,
            carry,
            call_actor_critic,
            key=key,
            eval_key=eval_key,
        )

        self.actor = solution.actor
        self.critic = solution.critic
        self.policy = solution.actor  # For base class compatibility
        self.history = aux.history
        # TODO do we need this variable?
        self.is_solved = True


@dataclasses.dataclass
class ActorCriticEnsembleSolver(BaseSolver[MDP]):
    """Scikit-learn style solver for actor-critic with critic ensembles."""

    obs_to_array: Callable = lambda obs: obs
    state_to_array: Callable = lambda state: state
    n_simulations: int = 32
    max_updates: int = 25_000
    max_env_steps: int | None = None
    td_lambda: float = 0.95
    critic_weight: float = 1.0
    optimism_coeff: float = 1.0
    n_ensemble_members: int = 4
    polyak_step_size: float | None = None

    # Optimizer configuration
    optax_optimizer: str = "adam"
    optax_optimizer_kws: dict = dataclasses.field(
        default_factory=lambda: {"learning_rate": 0.01}
    )
    optimizer_kws: dict = dataclasses.field(default_factory=dict)

    # Network configuration
    policy_mlp_kws: dict = dataclasses.field(
        default_factory=lambda: {"layer_sizes": [32]}
    )
    critic_mlp_kws: dict = dataclasses.field(
        default_factory=lambda: {"layer_sizes": [32]}
    )

    # Public attributes
    actor: Policy | None = dataclasses.field(init=False, default=None)
    critic: StateCritic | None = dataclasses.field(init=False, default=None)

    def _update_solution_attributes(self, solution):
        """Update actor, critic and policy with the solution."""
        self.actor = solution.actor
        self.critic = solution.critic
        self.policy = solution.actor

    def _prepare(
        self,
        problem: MDP,
        policy_init_key: JaxRandomKey,
        critic_init_keys: list[JaxRandomKey],
    ):
        """Initialize actor and critic ensemble networks."""
        sample_state = problem.init(key=policy_init_key)
        control_dim = problem.empty_control().shape[0]
        obs_dim = self.obs_to_array(sample_state).shape[0]
        state_dim = self.state_to_array(sample_state).shape[0]

        actor = _create_mlp_policy(
            obs_to_array=self.obs_to_array,
            obs_dim=obs_dim,
            control_dim=control_dim,
            key=policy_init_key,
            **self.policy_mlp_kws,
        )

        # Create ensemble of critics
        critics = []
        for _, critic_key in enumerate(critic_init_keys):
            critic = _create_mlp_critic(
                state_to_array=self.state_to_array,
                state_dim=state_dim,
                key=critic_key,
                **self.critic_mlp_kws,
            )
            critics.append(critic)

        critic_ensemble = tree_stack(critics)

        self.actor = actor
        self.critic = critic_ensemble
        self.policy = actor  # For base class compatibility
        self.prepared = True
        return actor, critic_ensemble

    def solve(
        self,
        problem: MDP,
        key: JaxRandomKey,
        eval_key: JaxRandomKey | None = None,
        policy_init_key: JaxRandomKey | None = None,
        critic_init_key: JaxRandomKey | None = None,
    ):
        """Solve the MDP using actor-critic with ensemble critics."""
        self.problem = problem

        if eval_key is None:
            key, eval_key = jr.split(key, 2)
        eval_key = cast(JaxRandomKey, eval_key)

        if policy_init_key is None:
            key, policy_init_key = jr.split(key, 2)
        policy_init_key = cast(JaxRandomKey, policy_init_key)

        if critic_init_key is None:
            key, critic_init_key = jr.split(key, 2)
        critic_init_key = cast(JaxRandomKey, critic_init_key)
        critic_init_keys = [
            i for i in jr.split(critic_init_key, self.n_ensemble_members)
        ]

        # Only prepare networks if not already prepared
        if not self.prepared:
            actor, critic_ensemble = self._prepare(
                problem, policy_init_key, critic_init_keys
            )
        else:
            actor = cast(Policy, self.actor)
            critic_ensemble = cast(StateCritic, self.critic)

        target_critic = (
            jax.tree.map(lambda x: x, critic_ensemble)
            if self.polyak_step_size is not None
            else None
        )

        optimizer = _create_optimizer(
            self.optax_optimizer, self.optax_optimizer_kws, self.optimizer_kws
        )

        actor_critic_optimizer = ActorCriticEnsembleOptimizer(
            mdp=problem,
            n_simulations=self.n_simulations,
            n_steps=self.steps_per_update,
            # The type below is not fixed yet here, since we will set the
            # objective to None. This is a flaw in our type system, which
            # needs to be fixed eventually.
            optimizer=optimizer,  # type: ignore
            critic_weight=self.critic_weight,
            td_lambda=self.td_lambda,
            optimism_coeff=self.optimism_coeff,
            steps_per_init=self.episode_length,
            polyak_step_size=self.polyak_step_size,
        )

        actor_critic = ActorCritic(
            actor=actor,
            critic=critic_ensemble,
            target_critic=target_critic,
        )
        carry = actor_critic_optimizer.initial_carry(
            sample_parameter=actor_critic
        )
        call_actor_critic = jax.jit(actor_critic_optimizer.__call__)

        solution, aux = self._run_training_loop(
            actor_critic_optimizer,
            carry,
            call_actor_critic,
            key=key,
            eval_key=eval_key,
        )

        self.actor = solution.actor
        self.critic = solution.critic
        self.policy = solution.actor  # For base class compatibility
        self.history = aux.history
        self.is_solved = True


@dataclasses.dataclass
class PolicyGradientSolver(BaseSolver[MDP]):
    """Policy-gradient-based estimation of Gaussian policies.

    Applicable to MDP instances.

    This solver implements policy gradient methods for continuous control tasks
    by optimizing stochastic policies that output Gaussian distribution
    parameters. Actions are sampled from tanh-transformed Gaussian
    distributions to ensure bounded controls that respect the MDP's action
    constraints.

    This is done using score function estimation, aka REINFORCE.

    The solver uses a two-level policy architecture:
    - Internal Gaussian policy: Outputs TanhGaussianPolicyControl parameters
    - External simulation policy: Samples concrete bounded actions for
      execution

    It is also possible to supply a return-dependent baseline for variance
    reduction.

    Attributes
    ----------
    obs_to_array : Callable
        Function to convert MDP observations to arrays for policy input.
    n_simulations : int
        Number of parallel rollouts per policy update.
    max_updates : int
        Maximum number of policy gradient updates to perform.
    optax_optimizer : str
        Name of the Optax optimizer to use (e.g., 'adam', 'sgd'). Used to
        access the optax module directly by string.
    optax_optimizer_kws : dict
        Keyword arguments for the Optax optimizer. Should be accepted by
        `optax.{optax_optimizer}`.
    policy_mlp_kws : dict
        Configuration for the policy MLP architecture.
    baseline_fn : Callable | None
        Optional baseline function for variance reduction in score function
        estimation.

    """

    obs_to_array: Callable = lambda obs: obs
    n_simulations: int = 32
    max_updates: int = 25_000
    max_env_steps: int | None = None

    # Optimizer configuration
    optax_optimizer: str = "adam"
    optax_optimizer_kws: dict = dataclasses.field(
        default_factory=lambda: {"learning_rate": 0.01}
    )
    optimizer_kws: dict = dataclasses.field(default_factory=dict)
    max_grad_norm: float | None = None

    # Policy configuration - outputs TanhGaussianPolicyControl
    policy_mlp_kws: dict = dataclasses.field(
        default_factory=lambda: {"layer_sizes": [32]}
    )

    # Baseline function for variance reduction
    baseline_fn: Callable | None = None

    def _update_solution_attributes(self, solution):
        """Update policy with the solution."""
        self._gaussian_policy = solution

        if self.problem is None:
            raise ValueError(".problem not set")

        self.policy = SimulationPolicy(
            solution, self.problem.control_min, self.problem.control_max
        )

    def _prepare(
        self,
        problem: MDP,
        policy_init_key: JaxRandomKey,
    ):
        """Initialize policy network that outputs TanhGaussianPolicyControl."""
        sample_state = problem.init(key=policy_init_key)
        control_dim = problem.empty_control().shape[0]
        obs_dim = self.obs_to_array(sample_state).shape[0]

        # Create policy that outputs loc and inv_softplus_scale
        # Output dimension is 2 * control_dim (loc + inv_softplus_scale)
        mlp_policy = _create_mlp_policy(
            obs_to_array=self.obs_to_array,
            obs_dim=obs_dim,
            control_dim=2 * control_dim,  # Double for loc + inv_softplus_scale
            key=policy_init_key,
            **self.policy_mlp_kws,
        )

        # Create array to control function
        def array_to_gaussian_control(arr):
            return TanhGaussianPolicyControl(
                loc=arr[:control_dim],
                inv_softplus_scale=arr[control_dim:],
            )

        # Create policy using StaticMLPPolicy directly
        gaussian_policy = StaticMLPPolicy(
            mlp=mlp_policy.mlp,
            obs_to_array=self.obs_to_array,
            array_to_control=array_to_gaussian_control,
        )

        # Create simulation policy that samples concrete actions
        self._gaussian_policy = gaussian_policy  # Store for training
        self.policy = SimulationPolicy(
            gaussian_policy, problem.control_min, problem.control_max
        )
        self.prepared = True
        return gaussian_policy

    def solve(
        self,
        problem: MDP,
        key: JaxRandomKey,
        eval_key: JaxRandomKey | None = None,
        policy_init_key: JaxRandomKey | None = None,
    ):
        """Solve the MDP using policy gradients."""
        self.problem = problem

        if eval_key is None:
            eval_key, key = jr.split(key)
            eval_key = cast(JaxRandomKey, eval_key)
        if policy_init_key is None:
            policy_init_key, key = jr.split(key)
            policy_init_key = cast(JaxRandomKey, policy_init_key)

        # Only prepare networks if not already prepared
        if not self.prepared:
            gaussian_policy = self._prepare(problem, policy_init_key)
        else:
            gaussian_policy = self._gaussian_policy

        optimizer = _create_optimizer(
            self.optax_optimizer,
            self.optax_optimizer_kws,
            self.optimizer_kws,
            max_grad_norm=self.max_grad_norm,
        )

        policy_gradient_optimizer = PolicyGradientOptimizer(
            mdp=problem,
            gaussian_policy=gaussian_policy,
            n_simulations=self.n_simulations,
            n_steps=self.steps_per_update,
            # The type below is not fixed yet here, since we will set the
            # objective to None. This is a flaw in our type system, which
            # needs to be fixed eventually.
            optimizer=optimizer,  # type: ignore
            baseline_fn=self.baseline_fn,
            steps_per_init=self.episode_length,
        )

        carry = policy_gradient_optimizer.initial_carry(
            sample_parameter=gaussian_policy
        )
        call_policy_gradient = jax.jit(policy_gradient_optimizer.__call__)

        solution, aux = self._run_training_loop(
            policy_gradient_optimizer,
            carry,
            call_policy_gradient,
            key=key,
            eval_key=eval_key,
        )

        # Update both the gaussian policy and simulation policy
        self._gaussian_policy = solution
        self.policy = SimulationPolicy(
            solution, problem.control_min, problem.control_max
        )
        self.history = aux.history
        self.is_solved = True

    def _run_training_loop(
        self,
        optimizer_instance,
        initial_carry,
        train_step_fn: Callable,
        key: JaxRandomKey,
        eval_key: JaxRandomKey,
    ):
        """Run training loop.

        This loop is adapted to deal with TanhGaussian MDPs.
        """
        del optimizer_instance

        if self.problem is None:
            raise ValueError(".problem not set")

        total_env_steps = 0

        carry = initial_carry
        for i in range(self.max_updates):
            key, search_key = jr.split(key, 2)
            carry, solution, aux = train_step_fn(carry, None, search_key)

            # Let subclass update its specific attributes
            self._update_solution_attributes(solution)

            total_env_steps += self.n_simulations * self.steps_per_update

            eval_history = None
            if (
                self.updates_per_eval is not None
                and i % self.updates_per_eval == 0
            ):
                eval_history = self.simulate(
                    n_steps=self.episode_length,
                    n_simulations=self.eval_n_simulations,
                    key=eval_key,
                )

            # Convert augmented train_history to original MDP format for
            # transparency.
            train_history = aux.history
            if train_history is not None:
                train_history = convert_tanh_gaussian_history_to_original(
                    train_history,
                    self.problem.control_min,
                    self.problem.control_max,
                )

            for callback in self.callbacks:
                callback(
                    i,
                    train_history=train_history,
                    eval_history=eval_history,
                    aux=aux,
                )

            if (
                self.max_env_steps is not None
                and total_env_steps >= self.max_env_steps
            ):
                break

        for callback in self.callbacks:
            if hasattr(callback, "teardown"):
                callback.teardown()

        return solution, aux


def plot_simulation_costs(
    history: History, height: int = 30, width: int = 80, show: bool = True
):
    """Plot simulation costs using plotext.

    Parameters
    ----------
    history
        History from solver.simulate() containing costs to plot.
    height
        Plot height in characters.
    width
        Plot width in characters.
    show
        Whether to display the plot. If False, only builds and returns plot.

    Returns
    -------
    Plot build result from plotext.build()

    """
    import plotext

    plotext.clf()
    plotext.plot_size(height=height, width=width)

    for i in range(history.costs.shape[0]):
        plotext.plot(history.costs[i].flatten().tolist())

    plotext.title("Simulation")
    plotext.xlabel("Environment Step")
    plotext.ylabel("Cost")

    if show:
        plotext.show()

    return plotext.build()
