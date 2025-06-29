"""Module holding various jax-specific utils."""

from dataclasses import is_dataclass
from typing import List, TypeVar

import jax
import jax.numpy as jnp

PyTree = TypeVar("PyTree")


def tree_stack(pytrees: List[PyTree]) -> PyTree:
    """Stack corresponding leaf arrays across a list of pytrees.

    This function takes multiple pytrees with identical structure and stacks
    corresponding leaf arrays along a new leading dimension. It handles regular
    nested structures (dicts, lists, tuples) as well as registered pytree nodes
    like dataclasses.

    Parameters
    ----------
    pytrees : List[PyTree]
        A list of pytrees with identical structure where each leaf is a JAX
        array. All corresponding leaf arrays must be compatible for stacking.

    Returns
    -------
    PyTree
        A single pytree with the same structure as the inputs, but where each
        leaf is a stacked array with shape (len(pytrees), *original_shape).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from flax.struct import dataclass
    >>>
    >>> # Example with dictionaries
    >>> pytree1 = {'a': jnp.array([1, 2]), 'b': {'c': jnp.array([3, 4])}}
    >>> pytree2 = {'a': jnp.array([5, 6]), 'b': {'c': jnp.array([7, 8])}}
    >>> stacked_dict = tree_stack([pytree1, pytree2])
    >>> print(stacked_dict['a'])
    [[1 2]
     [5 6]]
    >>>
    >>> # Example with dataclasses
    >>> @dataclass
    ... class Model:
    ...     weights: jnp.ndarray
    ...     bias: jnp.ndarray
    >>>
    >>> model1 = Model(weights=jnp.array([1.0, 2.0]), bias=jnp.array(0.5))
    >>> model2 = Model(weights=jnp.array([3.0, 4.0]), bias=jnp.array(1.5))
    >>> stacked_models = tree_stack([model1, model2])
    >>> print(stacked_models.weights)
    [[1. 2.]
     [3. 4.]]
    >>> print(stacked_models.bias)
    [0.5 1.5]

    Notes
    -----
    - All input pytrees must have identical structure.
    - When using with dataclasses, make sure to register them as pytree nodes
      using `jax.tree_util.register_pytree_node_class`.
    - If using a custom class, you must implement the `tree_flatten` and
      `tree_unflatten` methods and register it with JAX.

    See Also
    --------
    jax.tree_util.tree_map:
        The underlying function used for mapping across multiple trees.
    jnp.stack:
        The array stacking operation applied to each set of leaves.

    """
    if not pytrees:
        raise ValueError("Empty list provided. Must have at least one pytree.")

    # Function to stack corresponding leaves
    def stack_leaves(*leaves):
        return jnp.stack(leaves)

    # For standard pytrees, use tree_map directly
    if not is_dataclass(pytrees[0]):
        return jax.tree_util.tree_map(stack_leaves, *pytrees)

    # For dataclasses, we need to handle them specially
    # Get the dataclass structure from the first pytree
    first_tree = pytrees[0]
    flat_first, treedef = jax.tree_util.tree_flatten(first_tree)

    # Get the flattened representation of each pytree
    all_flattened = [jax.tree_util.tree_flatten(p)[0] for p in pytrees]

    # Stack corresponding leaves
    stacked_leaves = []
    for i in range(len(flat_first)):
        # Extract the i-th leaf from each flattened pytree
        corresponding_leaves = [flattened[i] for flattened in all_flattened]
        stacked_leaves.append(jnp.stack(corresponding_leaves))

    # Reconstruct with the original tree definition
    return jax.tree_util.tree_unflatten(treedef, stacked_leaves)
