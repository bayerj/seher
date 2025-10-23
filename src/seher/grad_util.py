"""Utilities for custom gradient operations."""

import jax


@jax.custom_vjp
def greater_than_ste(x: jax.Array, threshold: float) -> jax.Array:
    """Greater-than comparison with straight-through estimator gradient.

    In the forward pass, returns x > threshold.
    In the backward pass, uses straight-through estimation: gradients pass
    through unchanged.

    This is useful for constraint violations where we want to penalize only
    violations (x > threshold), but still want gradients to flow through to
    encourage the optimizer to reduce x.

    Parameters
    ----------
    x:
        Input array to compare against threshold.
    threshold:
        Threshold value for comparison.

    Returns
    -------
    Boolean array indicating where x > threshold.

    """
    return x > threshold


def greater_than_ste_fwd(x: jax.Array, threshold: float):
    """Forward pass for greater_than_ste."""
    return greater_than_ste(x, threshold), (x, threshold)


def greater_than_ste_bwd(res, g):
    """Backward pass with straight-through estimation."""
    x, threshold = res
    return (g, None)


greater_than_ste.defvjp(greater_than_ste_fwd, greater_than_ste_bwd)
