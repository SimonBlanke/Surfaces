# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Batch transformation utilities for BBOB and CEC benchmark functions.

This module provides vectorized versions of common transformations used
in benchmark optimization functions. All functions support array-backend
agnostic operations via get_array_namespace().

The transformations follow the BBOB/COCO definitions from:
    Hansen, N., et al. (2009). Real-parameter black-box optimization
    benchmarking 2009: Noiseless functions definitions.
"""

from surfaces._array_utils import ArrayLike, get_array_namespace


def batch_shift(X: ArrayLike, shift: ArrayLike) -> ArrayLike:
    """Apply shift transformation to batch of points.

    Computes z = x - o for each point.

    Parameters
    ----------
    X : ArrayLike
        Points, shape (n_points, n_dim).
    shift : ArrayLike
        Shift vector, shape (n_dim,).

    Returns
    -------
    ArrayLike
        Shifted points, shape (n_points, n_dim).
    """
    return X - shift  # Broadcasting handles this


def batch_rotate(X: ArrayLike, R: ArrayLike) -> ArrayLike:
    """Apply rotation transformation to batch of points.

    Computes z = R @ x for each point.

    Parameters
    ----------
    X : ArrayLike
        Points, shape (n_points, n_dim).
    R : ArrayLike
        Rotation matrix, shape (n_dim, n_dim).

    Returns
    -------
    ArrayLike
        Rotated points, shape (n_points, n_dim).
    """
    # For single point: z = R @ x
    # For batch: Z = X @ R.T (since (R @ x).T = x.T @ R.T)
    return X @ R.T


def batch_shift_rotate(X: ArrayLike, shift: ArrayLike, R: ArrayLike) -> ArrayLike:
    """Apply shift then rotation to batch of points.

    Computes z = R @ (x - o) for each point.

    Parameters
    ----------
    X : ArrayLike
        Points, shape (n_points, n_dim).
    shift : ArrayLike
        Shift vector, shape (n_dim,).
    R : ArrayLike
        Rotation matrix, shape (n_dim, n_dim).

    Returns
    -------
    ArrayLike
        Transformed points, shape (n_points, n_dim).
    """
    return (X - shift) @ R.T


def batch_t_osz(X: ArrayLike) -> ArrayLike:
    """Apply BBOB oscillation transformation T_osz to batch.

    Creates smooth oscillations around the identity to break symmetry.
    The transformation is defined as:

        T_osz(x) = sign(x) * exp(x_hat + 0.049 * (sin(c1*x_hat) + sin(c2*x_hat)))

    where:
        x_hat = log|x| (0 if x=0)
        c1 = 10 if x > 0 else 5.5
        c2 = 7.9 if x > 0 else 3.1

    Parameters
    ----------
    X : ArrayLike
        Points, shape (n_points, n_dim).

    Returns
    -------
    ArrayLike
        Transformed points, shape (n_points, n_dim).
    """
    xp = get_array_namespace(X)

    # Handle zeros: log(0) is undefined, use 0 instead
    x_hat = xp.where(X != 0, xp.log(xp.abs(X)), 0.0)

    # Different coefficients for positive vs negative values
    c1 = xp.where(X > 0, 10.0, 5.5)
    c2 = xp.where(X > 0, 7.9, 3.1)

    # Compute transformation
    result = xp.sign(X) * xp.exp(x_hat + 0.049 * (xp.sin(c1 * x_hat) + xp.sin(c2 * x_hat)))

    # Where X was 0, keep it 0
    return xp.where(X != 0, result, 0.0)


def batch_t_asy(X: ArrayLike, beta: float, n_dim: int = None) -> ArrayLike:
    """Apply BBOB asymmetry transformation T_asy^beta to batch.

    Breaks symmetry by applying different scaling to positive values.
    The transformation is defined as:

        T_asy(x_i) = x_i^(1 + beta * i/(n-1) * sqrt(x_i))  if x_i > 0
                   = x_i                                    otherwise

    Parameters
    ----------
    X : ArrayLike
        Points, shape (n_points, n_dim).
    beta : float
        Asymmetry parameter (typically 0.2 or 0.5).
    n_dim : int, optional
        Number of dimensions. If None, inferred from X.shape[1].

    Returns
    -------
    ArrayLike
        Transformed points, shape (n_points, n_dim).
    """
    xp = get_array_namespace(X)

    if n_dim is None:
        n_dim = X.shape[1]

    # Create index array [0, 1, 2, ..., n_dim-1]
    i = xp.arange(n_dim, dtype=X.dtype)

    # Compute exponent: 1 + beta * i / (n_dim - 1) * sqrt(max(x, 0))
    if n_dim > 1:
        exp = 1 + beta * i / (n_dim - 1) * xp.sqrt(xp.maximum(X, 0))
    else:
        exp = 1 + beta * xp.sqrt(xp.maximum(X, 0))

    # Apply only where X > 0
    return xp.where(X > 0, xp.power(X, exp), X)


def batch_f_pen(X: ArrayLike, bound: float = 5.0) -> ArrayLike:
    """Compute BBOB boundary penalty for batch of points.

    Penalizes solutions outside [-bound, bound]^D.

    Parameters
    ----------
    X : ArrayLike
        Points, shape (n_points, n_dim).
    bound : float, default=5.0
        Boundary value.

    Returns
    -------
    ArrayLike
        Penalty values, shape (n_points,).
    """
    xp = get_array_namespace(X)
    return xp.sum(xp.maximum(0, xp.abs(X) - bound) ** 2, axis=1)


def batch_lambda_alpha(X: ArrayLike, alpha: float, n_dim: int = None) -> ArrayLike:
    """Apply diagonal conditioning matrix Lambda^alpha to batch.

    Multiplies each dimension by alpha^(0.5 * i / (n-1)).

    Parameters
    ----------
    X : ArrayLike
        Points, shape (n_points, n_dim).
    alpha : float
        Condition number (ratio of largest to smallest scaling).
    n_dim : int, optional
        Number of dimensions. If None, inferred from X.shape[1].

    Returns
    -------
    ArrayLike
        Scaled points, shape (n_points, n_dim).
    """
    xp = get_array_namespace(X)

    if n_dim is None:
        n_dim = X.shape[1]

    # Create diagonal scaling factors
    i = xp.arange(n_dim, dtype=X.dtype)
    if n_dim > 1:
        diag = xp.power(alpha, 0.5 * i / (n_dim - 1))
    else:
        diag = xp.ones(1, dtype=X.dtype)

    # Apply element-wise (equivalent to X @ diag_matrix for diagonal matrix)
    return X * diag
