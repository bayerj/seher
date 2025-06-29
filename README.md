# seher

Towards model-based reinforcement learning with jax.

Some guiding principles.

 - Write Python like it's rust.
 - Rely on protocols/jax-transformable dataclasses as much as possible,
   keep implicit semantics of array dimensions minimal.
 - Everything is type annotated and tested.
 - No explicit parallelization, all done with jax primitives.
 - Maximum reuse of already existing tooling such as equinox, optax, etc.

Some features.

 - MPC with a derivative-free optimizer, MPPI.
 - Policy search and actor critic.
 - Interface to mujoco_playground.


## Roadmap

- Milestone "Stochastic Policies"
  - [ ] Make it more generic such that it is not only about Gaussian policies.
- Milestone "Maintenance"
  - [ ] Add a layer of abstraction that makes it easier to write performance
        tests, since these have a tendency to be invalidate after huge chunks
        of work.
- Milestone "Features"
  - MPPI
    - [ ] respect control spaces (fit a truncated gaussian)
- Milestone "Stepper"
  - [ ] Find a way that using the specific carries in subclasses of Stepper
        classes can be used, and type checkers don't complain. Right now, we
        need to use the super class in the signature, and then cast to the
        subclass in the method.
  - [ ] Make `problem_data` optional for optimizers
  - [ ] add an option to also vmap the random keys during Optimizer steps
- Milestone "Augmented Random Search"
  - [ ] tune ARS-MPC on the walker_walk, cheating on the obs normalizations
  - [ ] check if the symlog transformation works well on walkerwalk, so no
        normalization needs to be implemented right away
  - [ ] decide for good how to return auxiliary data 
- Milestone "Quality"
  - [ ] Make the initial state show up in histories, currently it does not.
  - [ ] Make sure histories have the right shapes (can be done with the one
        before this)
  - [ ] Make RichProgressCallback not based on policies (the variable names
        indicate it)
  - [ ] The seher.control.{actor_critic,policy_search} modules need some
        unification, they have different architectures. also, some functions
        from the latter are used in the former.
  - [ ] Remove double sampling in TanhGaussianPolicyMDP by ensuring cost()
        uses the same control sample as transit() instead of sampling twice
- Milestone "UX"
  - [x] Add plots to cli callback
  - [x] Make cli callback plots have the proper x axis of amount of
        iterations, instead of the rather arbitrary 1000.
  - [ ] Fix those plots. I think I fucked them up big time, they don't really
        plot nicely. The appending is too magical. I think I need to append
        everything and then interpolate. Don't do smart stuff, Justin!
- Milestone "Policy search"
  - [x] Implement a variant that starts from states of the previous iteration,
        so that we can do updates every 10 steps or so, but run envs for
        500 steps (like in standard DMC)
  - [x] Make that work for ARS as well, which cannot do value_and_grad with
        aux
  - Make it work on WalkerWalk
    - Parametrize policy search script
      - [ ] pick dmc env
      - [ ] run study or individual experiment
      - [ ] make costs.txt callback
      - [x] remove obs hard code in favour of symlog
      - [ ] properly calc all these steps and update ratios
- Milestone "Actor-Critic"
  - [ ] undo relaxation of performance tests in 6bcaa44.
  - [ ] implement a variant that can use ARS for mujoco_playground
        environments; here is a breakdown
        - refactor actor critic such that it has a policy objective and
          a crtic objective.
        - the former produces a history as its auxiliary and this can then
          be passed to the latter
        - both also get two different optimizers which produce steps--here
          the policy can get an ARS-based optimiser and the critic one using
          ordinary gradients
- Milestone "Constrained control"
  - [ ] Make a stepper that can work with cost and constraints
  - [ ] Add a simple constrained mdp, e.g. point in a box or so
  - PolicySearch
     - [ ] Add a stepper that trains a policy on constrained MDPs
     - [ ] Make it work and add a performance test!
  - MPC
     - [ ] Extend MPC with constraints
     - [ ] Make it work and add a performance test!
- Milestone "Partial observability"
  - [ ] Add a StateEstimator/Filter protocol
  - [ ] Add a challenging but simple POMDP
  - [ ] Add/extend the simulate function to work with POMDPs and state
        estimators
  - PolicySearch
     - [ ] Add a stepper that trains a recurrent policy
     - [ ] Make it work and add a performance test!
  - MPC
     - [ ] Extend MPC with Filters
     - [ ] Make it work and add a performance test!
- Milestone "Newton RL"
- Milestone "Model learning"
- Milestone "Adaptive control"
- Milestone "Maximum entropy control"
