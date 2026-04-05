# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""WFG transformation primitives.

These pure functions implement the bias, shift, reduction, and shape
transformations defined in the WFG toolkit. Each operates on numpy arrays
and clips outputs to [0, 1] where appropriate to prevent numerical drift.

Reference: Huband, S., Hingston, P., Barone, L., & While, L. (2006).
A review of multiobjective test problems and a scalable test problem toolkit.
IEEE Transactions on Evolutionary Computation, 10(5), 477-506.
"""

import numpy as np


def b_poly(y, alpha):
    """Polynomial bias: y^alpha."""
    return np.clip(y, 0, 1) ** alpha


def b_flat(y, A, B, C):
    """Flat region bias.

    Produces a flat region at value A for y in [B, C]. Outside that
    interval the function ramps linearly toward A from both sides.
    """
    tmp1 = np.minimum(0.0, np.floor(y - B)) * A * (B - y) / B
    tmp2 = np.minimum(0.0, np.floor(C - y)) * (1 - A) * (y - C) / (1 - C)
    return np.clip(A + tmp1 - tmp2, 0, 1)


def b_param(y, u, A, B, C):
    """Parameter-dependent bias.

    The exponent varies with the auxiliary value u, creating a bias
    whose strength depends on other variables in the problem.
    """
    v = A - (1 - 2 * u) * np.abs(np.floor(0.5 - u) + A)
    return np.clip(y ** (B + (C - B) * v), 0, 1)


def s_linear(y, A):
    """Linear shift: |y - A| / max(A, 1-A)."""
    return np.clip(np.abs(y - A) / np.maximum(A, 1 - A), 0, 1)


def s_deceptive(y, A, B, C):
    """Deceptive shift with local optima.

    Creates a landscape with a global optimum at A and deceptive
    local optima that can mislead optimizers.
    """
    tmp1 = np.floor(y - A + B) * (1 - C + (A - B) / B) / (A - B)
    tmp2 = np.floor(A + B - y) * (1 - C + (1 - A - B) / B) / (1 - A - B)
    return np.clip(1 + (np.abs(y - A) - B) * (tmp1 + tmp2 + 1 / B), 0, 1)


def s_multi_modal(y, A, B, C):
    """Multi-modal shift with A peaks.

    Combines a cosine component for multiple peaks with a quadratic
    envelope that makes the global optimum at C dominant.
    """
    tmp1 = np.abs(y - C) / (2 * (np.floor(C - y) + C))
    tmp2 = (4 * A + 2) * np.pi * (0.5 - tmp1)
    return np.clip((1 + np.cos(tmp2) + 4 * B * tmp1**2) / (B + 2), 0, 1)


def r_sum(y, w):
    """Weighted sum reduction to scalar in [0,1]."""
    return np.dot(w, y) / np.sum(w)


def r_nonsep(y, A):
    """Non-separable reduction.

    A controls the degree of non-separability: A=1 is fully separable,
    A=len(y) is fully non-separable. The modular indexing creates
    coupling between adjacent variables.
    """
    n = len(y)
    val = 0.0
    for j in range(n):
        val += y[j]
        for kk in range(int(A) - 1):
            val += np.abs(y[j] - y[(j + 1 + kk) % n])
    denom = n * np.ceil(A / 2.0) * (1.0 + 2 * A - 2 * np.ceil(A / 2.0)) / A
    return val / denom


def convex(x, M, m):
    """Convex Pareto front shape function.

    Parameters
    ----------
    x : np.ndarray
        Array of length M-1 with values in [0, 1].
    M : int
        Total number of objectives.
    m : int
        Current objective index (0-indexed).
    """
    if m == 0:
        return np.prod(1 - np.cos(x * np.pi / 2))
    elif m < M - 1:
        return np.prod(1 - np.cos(x[: M - 1 - m] * np.pi / 2)) * (
            1 - np.sin(x[M - 1 - m] * np.pi / 2)
        )
    else:
        return 1 - np.sin(x[0] * np.pi / 2)


def concave(x, M, m):
    """Concave Pareto front shape function."""
    if m == 0:
        return np.prod(np.sin(x * np.pi / 2))
    elif m < M - 1:
        return np.prod(np.sin(x[: M - 1 - m] * np.pi / 2)) * np.cos(x[M - 1 - m] * np.pi / 2)
    else:
        return np.cos(x[0] * np.pi / 2)


def linear(x, M, m):
    """Linear Pareto front shape function."""
    if m == 0:
        return np.prod(x)
    elif m < M - 1:
        return np.prod(x[: M - 1 - m]) * (1 - x[M - 1 - m])
    else:
        return 1 - x[0]


def mixed(x, A, alpha):
    """Mixed convex/concave shape. Uses only x[0]."""
    return (1 - x[0] - np.cos(2 * A * np.pi * x[0] + np.pi / 2) / (2 * A * np.pi)) ** alpha


def disconnected(x, A, alpha, beta):
    """Disconnected regions shape. Uses only x[0]."""
    return 1 - x[0] ** alpha * np.cos(A * x[0] ** beta * np.pi) ** 2
