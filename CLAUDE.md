# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Testing
```bash
# Quick tests (parallel, excluding slow tests)
rye run quick-test

# Quick tests excluding performance tests
rye run quick-test-no-perf  

# All quick tests (non-parallel)
rye run quick-test-all

# Full test suite
rye run test

# Run specific test
pytest tests/seher/control/test_actor_critic.py -v
```

### Type Checking & Linting
```bash
# Type checking with Pyright
rye run pyright-run

# Linting and formatting
ruff check --fix src/
ruff format src/
```

### Examples
Policy search examples are in `examples/dmc/` and use Optuna for hyperparameter tuning.

## Architecture Notes

- The codebase uses JAX extensively for automatic differentiation and JIT compilation
- All array operations should be JAX-compatible
- State and control are typically represented as JAX arrays
- Callbacks are used extensively for logging and visualization (e.g., `MCPSCallback`, `MCPSCLIProgressCallback`)
- The `seher.types` module contains common type definitions
- Performance tests are marked with `@pytest.mark.performance` and can be skipped during development
- Use the following JAX import conventions:
  - `import jax.tree as jt`
  - `import jax.random as jr`
  - `import jax.nn as jnn`
  - `import jax.numpy as jnp`

## Environment Setup

This project uses Rye for dependency management. The main dependencies include JAX, dm-control, MuJoCo, Optax, and various visualization libraries.

## Code Style Guidelines

- Put all imports to the top of the file.
- Make comments full sentences and end them with a full stop.
- Only add comments to explain why it does something non-obvious.
- Check the pyproject.toml file for formatting rules, especially for line limits.
- Follow numpydoc documentation.
