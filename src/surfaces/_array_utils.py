# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Array utilities for backend-agnostic batch evaluation.

This module provides utilities for working with different array backends
(numpy, cupy, jax) in a unified way. The key function is get_array_namespace()
which detects the array library and returns the appropriate module.

Examples
--------
>>> import numpy as np
>>> X = np.array([[1.0, 2.0], [3.0, 4.0]])
>>> xp = get_array_namespace(X)
>>> xp.sum(X**2, axis=1)
array([ 5., 25.])
"""

from typing import Any, Protocol, Tuple, runtime_checkable


@runtime_checkable
class ArrayLike(Protocol):
    """Protocol for NumPy-compatible array objects.

    This protocol defines the minimal interface expected from array objects.
    Compatible with: numpy.ndarray, cupy.ndarray, jax.numpy.ndarray.

    We expect NumPy-compatible API semantics. If an exotic library has a
    different API, adapter classes are the user's responsibility.
    """

    shape: Tuple[int, ...]
    ndim: int
    dtype: Any

    def __getitem__(self, key: Any) -> Any: ...


def get_array_namespace(x: ArrayLike) -> Any:
    """Detect array library and return the appropriate module.

    This function enables backend-agnostic array operations by detecting
    which array library created the input and returning that library's
    module for further operations.

    Parameters
    ----------
    x : ArrayLike
        An array-like object (numpy, cupy, jax array).

    Returns
    -------
    module
        The array module (numpy, cupy, or jax.numpy).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1.0, 2.0]])
    >>> xp = get_array_namespace(X)
    >>> xp is np
    True

    Notes
    -----
    Detection order:
    1. Array API Standard (__array_namespace__) - Python 3.11+, numpy 2.0+
    2. Module name detection - fallback for older versions
    3. Default to numpy if unknown
    """
    # Array API Standard (PEP 706) - modern arrays support this
    if hasattr(x, "__array_namespace__"):
        return x.__array_namespace__()

    # Fallback: detect by module name
    module_name = type(x).__module__.split(".")[0]

    if module_name == "cupy":
        import cupy

        return cupy

    if module_name == "jax":
        import jax.numpy

        return jax.numpy

    # Default to numpy (works for numpy.ndarray and most compatible arrays)
    import numpy

    return numpy


def is_array_like(x: Any) -> bool:
    """Check if x is an array-like object suitable for batch evaluation.

    Parameters
    ----------
    x : Any
        Object to check.

    Returns
    -------
    bool
        True if x has shape, ndim, dtype attributes and is indexable.
    """
    return (
        hasattr(x, "shape")
        and hasattr(x, "ndim")
        and hasattr(x, "dtype")
        and hasattr(x, "__getitem__")
    )
