"""Compute Units (CU) for hardware-independent cost measurement.

1 CU equals the time of one reference operation on the current machine.
The reference operation mixes arithmetic, transcendental, and matrix-vector
operations to represent the computational profile of different test function
categories.

Usage within Surfaces:
    Each test function has a pre-computed ``eval_cost`` attribute in its
    ``_spec`` dict, expressed in CU.

Usage for benchmarking (external):
    Use ``to_cu()`` to convert wall-clock seconds to CU, enabling
    direct comparison of function eval cost and optimizer overhead.

    >>> from surfaces.cost import to_cu
    >>> optimizer_cu = to_cu(measured_optimizer_seconds)
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

_ref_time: Optional[float] = None


def calibrate(min_duration: float = 1.0) -> float:
    """Measure seconds per reference operation.

    The reference operation combines arithmetic, transcendental, and
    matrix-vector operations to represent the computational mix of
    different test function categories (algebraic, BBOB, CEC, ML).

    Results are cached module-level. Call ``reset()`` to re-calibrate.

    Parameters
    ----------
    min_duration : float, default=1.0
        Minimum duration of the main measurement in seconds.
        Longer durations reduce measurement noise.

    Returns
    -------
    float
        Seconds per reference operation.
    """
    global _ref_time
    if _ref_time is not None:
        return _ref_time

    rng = np.random.RandomState(42)
    n = 50

    x_vec = rng.randn(n)
    x_mat = rng.randn(n, n)
    x_exp = rng.randn(n) * 0.01

    def ref_op():
        a = np.sum(x_vec * x_vec)
        b = np.sum(np.sin(x_vec)) + np.sum(np.exp(x_exp))
        c = x_mat @ x_vec
        return a + b + np.sum(c)

    # Warmup: CPU caches, numpy internals
    for _ in range(2000):
        ref_op()

    # Pilot measurement to estimate time per operation
    n_pilot = 5000
    t0 = time.perf_counter()
    for _ in range(n_pilot):
        ref_op()
    t_pilot = time.perf_counter() - t0
    time_per_op = t_pilot / n_pilot

    # Compute iteration count to reach min_duration
    n_iter = max(10_000, int(min_duration / time_per_op))

    # Main measurement
    t0 = time.perf_counter()
    for _ in range(n_iter):
        ref_op()
    elapsed = time.perf_counter() - t0

    _ref_time = elapsed / n_iter
    return _ref_time


def to_cu(seconds: float) -> float:
    """Convert wall-clock seconds to Compute Units.

    Auto-calibrates on first call if not yet calibrated.

    Parameters
    ----------
    seconds : float
        Wall-clock time in seconds.

    Returns
    -------
    float
        Equivalent cost in Compute Units (CU).
    """
    return seconds / calibrate()


def reset() -> None:
    """Clear the cached calibration value.

    Call this to force re-calibration on the next ``calibrate()``
    or ``to_cu()`` call.
    """
    global _ref_time
    _ref_time = None
