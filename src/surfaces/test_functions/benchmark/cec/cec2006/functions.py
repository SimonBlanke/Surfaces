# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2006 constrained benchmark functions G01-G24.

These functions are from the CEC 2006 Special Session on Constrained
Real-Parameter Optimization. Each function has a fixed dimension and
various types of constraints (linear/nonlinear, inequality/equality).

References
----------
Liang, J. J., et al. (2006). Problem definitions and evaluation criteria
for the CEC 2006 special session on constrained real-parameter optimization.
Technical Report, Nanyang Technological University, Singapore.
"""

from typing import List, Optional

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from ._base_cec2006 import CEC2006Function


# =============================================================================
# G01: 13-dim, 9 linear inequality constraints
# =============================================================================


class G01(CEC2006Function):
    """G01: Quadratic objective with 9 linear inequality constraints.

    Dimension: 13
    Constraints: 9 linear inequalities
    Optimal: f* = -15.0 at x* = (1,1,1,1,1,1,1,1,1,3,3,3,1)
    """

    _spec = {"func_id": 1, "name": "G01"}
    _n_dim = 13
    _n_linear_ineq = 9
    _n_nonlinear_eq = 0
    _n_nonlinear_ineq = 0
    _f_global = -15.0
    _x_global = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1], dtype=np.float64)
    _variable_bounds = [(0, 1)] * 9 + [(0, 100)] * 3 + [(0, 1)]

    def raw_objective(self, x: np.ndarray) -> float:
        return (
            5 * np.sum(x[:4])
            - 5 * np.sum(x[:4] ** 2)
            - np.sum(x[4:13])
        )

    def inequality_constraints(self, x: np.ndarray) -> List[float]:
        g = []
        g.append(2 * x[0] + 2 * x[1] + x[9] + x[10] - 10)
        g.append(2 * x[0] + 2 * x[2] + x[9] + x[11] - 10)
        g.append(2 * x[1] + 2 * x[2] + x[10] + x[11] - 10)
        g.append(-8 * x[0] + x[9])
        g.append(-8 * x[1] + x[10])
        g.append(-8 * x[2] + x[11])
        g.append(-2 * x[3] - x[4] + x[9])
        g.append(-2 * x[5] - x[6] + x[10])
        g.append(-2 * x[7] - x[8] + x[11])
        return g

    def _batch_raw_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        return (
            5 * xp.sum(X[:, :4], axis=1)
            - 5 * xp.sum(X[:, :4] ** 2, axis=1)
            - xp.sum(X[:, 4:13], axis=1)
        )

    def _batch_inequality_constraints(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n = X.shape[0]
        G = xp.zeros((n, 9))
        G[:, 0] = 2 * X[:, 0] + 2 * X[:, 1] + X[:, 9] + X[:, 10] - 10
        G[:, 1] = 2 * X[:, 0] + 2 * X[:, 2] + X[:, 9] + X[:, 11] - 10
        G[:, 2] = 2 * X[:, 1] + 2 * X[:, 2] + X[:, 10] + X[:, 11] - 10
        G[:, 3] = -8 * X[:, 0] + X[:, 9]
        G[:, 4] = -8 * X[:, 1] + X[:, 10]
        G[:, 5] = -8 * X[:, 2] + X[:, 11]
        G[:, 6] = -2 * X[:, 3] - X[:, 4] + X[:, 9]
        G[:, 7] = -2 * X[:, 5] - X[:, 6] + X[:, 10]
        G[:, 8] = -2 * X[:, 7] - X[:, 8] + X[:, 11]
        return G


# =============================================================================
# G02: 20-dim, 2 nonlinear inequality constraints
# =============================================================================


class G02(CEC2006Function):
    """G02: Nonlinear objective with 2 nonlinear inequality constraints.

    Dimension: 20
    Constraints: 2 nonlinear inequalities
    Optimal: f* = -0.803619 (approximate)
    """

    _spec = {"func_id": 2, "name": "G02"}
    _n_dim = 20
    _n_linear_ineq = 0
    _n_nonlinear_eq = 0
    _n_nonlinear_ineq = 2
    _f_global = -0.803619
    _x_global = None  # No closed-form solution
    _variable_bounds = [(0, 10)] * 20

    def raw_objective(self, x: np.ndarray) -> float:
        n = len(x)
        i = np.arange(1, n + 1)
        cos_sum = np.sum(np.cos(x) ** 4)
        cos_prod = np.prod(np.cos(x) ** 2)
        sqrt_term = np.sqrt(np.sum(i * x**2))
        if sqrt_term == 0:
            return 0
        return -np.abs((cos_sum - 2 * cos_prod) / sqrt_term)

    def inequality_constraints(self, x: np.ndarray) -> List[float]:
        g = []
        g.append(0.75 - np.prod(x))
        g.append(np.sum(x) - 7.5 * len(x))
        return g

    def _batch_raw_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n = X.shape[1]
        i = xp.arange(1, n + 1, dtype=X.dtype)
        cos_sum = xp.sum(xp.cos(X) ** 4, axis=1)
        cos_prod = xp.prod(xp.cos(X) ** 2, axis=1)
        sqrt_term = xp.sqrt(xp.sum(i * X**2, axis=1))
        result = xp.where(
            sqrt_term == 0,
            0.0,
            -xp.abs((cos_sum - 2 * cos_prod) / sqrt_term)
        )
        return result

    def _batch_inequality_constraints(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n = X.shape[0]
        G = xp.zeros((n, 2))
        G[:, 0] = 0.75 - xp.prod(X, axis=1)
        G[:, 1] = xp.sum(X, axis=1) - 7.5 * X.shape[1]
        return G


# =============================================================================
# G03: 10-dim, 1 nonlinear equality constraint
# =============================================================================


class G03(CEC2006Function):
    """G03: Polynomial objective with 1 nonlinear equality constraint.

    Dimension: 10
    Constraints: 1 nonlinear equality
    Optimal: f* = -1.0 at x* = (1/sqrt(n), ..., 1/sqrt(n))
    """

    _spec = {"func_id": 3, "name": "G03"}
    _n_dim = 10
    _n_linear_ineq = 0
    _n_nonlinear_eq = 1
    _n_nonlinear_ineq = 0
    _f_global = -1.0
    _x_global = np.ones(10) / np.sqrt(10)
    _variable_bounds = [(0, 1)] * 10

    def raw_objective(self, x: np.ndarray) -> float:
        n = len(x)
        return -(np.sqrt(n) ** n) * np.prod(x)

    def equality_constraints(self, x: np.ndarray) -> List[float]:
        return [np.sum(x**2) - 1]

    def _batch_raw_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n = X.shape[1]
        return -(np.sqrt(n) ** n) * xp.prod(X, axis=1)

    def _batch_equality_constraints(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        return xp.sum(X**2, axis=1, keepdims=True) - 1


# =============================================================================
# G04: 5-dim, 6 nonlinear inequality constraints
# =============================================================================


class G04(CEC2006Function):
    """G04: Quadratic objective with 6 nonlinear inequality constraints.

    Dimension: 5
    Constraints: 6 nonlinear inequalities
    Optimal: f* = -30665.539
    """

    _spec = {"func_id": 4, "name": "G04"}
    _n_dim = 5
    _n_linear_ineq = 0
    _n_nonlinear_eq = 0
    _n_nonlinear_ineq = 6
    _f_global = -30665.539
    _x_global = np.array([78, 33, 29.9952560256816, 45, 36.7758129057882])
    _variable_bounds = [(78, 102), (33, 45), (27, 45), (27, 45), (27, 45)]

    def raw_objective(self, x: np.ndarray) -> float:
        return (
            5.3578547 * x[2]**2
            + 0.8356891 * x[0] * x[4]
            + 37.293239 * x[0]
            - 40792.141
        )

    def inequality_constraints(self, x: np.ndarray) -> List[float]:
        u = (
            85.334407
            + 0.0056858 * x[1] * x[4]
            + 0.0006262 * x[0] * x[3]
            - 0.0022053 * x[2] * x[4]
        )
        v = (
            80.51249
            + 0.0071317 * x[1] * x[4]
            + 0.0029955 * x[0] * x[1]
            + 0.0021813 * x[2]**2
        )
        w = (
            9.300961
            + 0.0047026 * x[2] * x[4]
            + 0.0012547 * x[0] * x[2]
            + 0.0019085 * x[2] * x[3]
        )
        g = []
        g.append(u - 92)
        g.append(-u)
        g.append(v - 110)
        g.append(90 - v)
        g.append(w - 25)
        g.append(20 - w)
        return g

    def _batch_raw_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        return (
            5.3578547 * X[:, 2]**2
            + 0.8356891 * X[:, 0] * X[:, 4]
            + 37.293239 * X[:, 0]
            - 40792.141
        )

    def _batch_inequality_constraints(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        u = (
            85.334407
            + 0.0056858 * X[:, 1] * X[:, 4]
            + 0.0006262 * X[:, 0] * X[:, 3]
            - 0.0022053 * X[:, 2] * X[:, 4]
        )
        v = (
            80.51249
            + 0.0071317 * X[:, 1] * X[:, 4]
            + 0.0029955 * X[:, 0] * X[:, 1]
            + 0.0021813 * X[:, 2]**2
        )
        w = (
            9.300961
            + 0.0047026 * X[:, 2] * X[:, 4]
            + 0.0012547 * X[:, 0] * X[:, 2]
            + 0.0019085 * X[:, 2] * X[:, 3]
        )
        n = X.shape[0]
        G = xp.zeros((n, 6))
        G[:, 0] = u - 92
        G[:, 1] = -u
        G[:, 2] = v - 110
        G[:, 3] = 90 - v
        G[:, 4] = w - 25
        G[:, 5] = 20 - w
        return G


# =============================================================================
# G05: 4-dim, 2 linear inequality, 3 nonlinear equality constraints
# =============================================================================


class G05(CEC2006Function):
    """G05: Cubic objective with mixed constraints.

    Dimension: 4
    Constraints: 2 linear inequalities, 3 nonlinear equalities
    Optimal: f* = 5126.4981
    """

    _spec = {"func_id": 5, "name": "G05"}
    _n_dim = 4
    _n_linear_ineq = 2
    _n_nonlinear_eq = 3
    _n_nonlinear_ineq = 0
    _f_global = 5126.4981
    _x_global = np.array([679.9453, 1026.067, 0.1188764, -0.3962336])
    _variable_bounds = [(0, 1200), (0, 1200), (-0.55, 0.55), (-0.55, 0.55)]

    def raw_objective(self, x: np.ndarray) -> float:
        return 3 * x[0] + 0.000001 * x[0]**3 + 2 * x[1] + 0.000002 / 3 * x[1]**3

    def inequality_constraints(self, x: np.ndarray) -> List[float]:
        g = []
        g.append(-x[3] + x[2] - 0.55)
        g.append(-x[2] + x[3] - 0.55)
        return g

    def equality_constraints(self, x: np.ndarray) -> List[float]:
        h = []
        h.append(1000 * np.sin(-x[2] - 0.25) + 1000 * np.sin(-x[3] - 0.25) + 894.8 - x[0])
        h.append(1000 * np.sin(x[2] - 0.25) + 1000 * np.sin(x[2] - x[3] - 0.25) + 894.8 - x[1])
        h.append(1000 * np.sin(x[3] - 0.25) + 1000 * np.sin(x[3] - x[2] - 0.25) + 1294.8)
        return h

    def _batch_raw_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        return 3 * X[:, 0] + 0.000001 * X[:, 0]**3 + 2 * X[:, 1] + 0.000002 / 3 * X[:, 1]**3

    def _batch_inequality_constraints(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n = X.shape[0]
        G = xp.zeros((n, 2))
        G[:, 0] = -X[:, 3] + X[:, 2] - 0.55
        G[:, 1] = -X[:, 2] + X[:, 3] - 0.55
        return G

    def _batch_equality_constraints(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n = X.shape[0]
        H = xp.zeros((n, 3))
        H[:, 0] = 1000 * xp.sin(-X[:, 2] - 0.25) + 1000 * xp.sin(-X[:, 3] - 0.25) + 894.8 - X[:, 0]
        H[:, 1] = 1000 * xp.sin(X[:, 2] - 0.25) + 1000 * xp.sin(X[:, 2] - X[:, 3] - 0.25) + 894.8 - X[:, 1]
        H[:, 2] = 1000 * xp.sin(X[:, 3] - 0.25) + 1000 * xp.sin(X[:, 3] - X[:, 2] - 0.25) + 1294.8
        return H


# =============================================================================
# G06: 2-dim, 2 nonlinear inequality constraints
# =============================================================================


class G06(CEC2006Function):
    """G06: Cubic objective with 2 nonlinear inequality constraints.

    Dimension: 2
    Constraints: 2 nonlinear inequalities
    Optimal: f* = -6961.81388
    """

    _spec = {"func_id": 6, "name": "G06"}
    _n_dim = 2
    _n_linear_ineq = 0
    _n_nonlinear_eq = 0
    _n_nonlinear_ineq = 2
    _f_global = -6961.81388
    _x_global = np.array([14.095, 0.84296])
    _variable_bounds = [(13, 100), (0, 100)]

    def raw_objective(self, x: np.ndarray) -> float:
        return (x[0] - 10)**3 + (x[1] - 20)**3

    def inequality_constraints(self, x: np.ndarray) -> List[float]:
        g = []
        g.append(-(x[0] - 5)**2 - (x[1] - 5)**2 + 100)
        g.append((x[0] - 6)**2 + (x[1] - 5)**2 - 82.81)
        return g

    def _batch_raw_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        return (X[:, 0] - 10)**3 + (X[:, 1] - 20)**3

    def _batch_inequality_constraints(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n = X.shape[0]
        G = xp.zeros((n, 2))
        G[:, 0] = -(X[:, 0] - 5)**2 - (X[:, 1] - 5)**2 + 100
        G[:, 1] = (X[:, 0] - 6)**2 + (X[:, 1] - 5)**2 - 82.81
        return G


# =============================================================================
# G07: 10-dim, 3 linear inequality, 5 nonlinear inequality constraints
# =============================================================================


class G07(CEC2006Function):
    """G07: Quadratic objective with 8 inequality constraints.

    Dimension: 10
    Constraints: 3 linear + 5 nonlinear inequalities
    Optimal: f* = 24.3062091
    """

    _spec = {"func_id": 7, "name": "G07"}
    _n_dim = 10
    _n_linear_ineq = 3
    _n_nonlinear_eq = 0
    _n_nonlinear_ineq = 5
    _f_global = 24.3062091
    _x_global = np.array([
        2.171996, 2.363683, 8.773926, 5.095984, 0.9906548,
        1.430574, 1.321644, 9.828726, 8.280092, 8.375927
    ])
    _variable_bounds = [(-10, 10)] * 10

    def raw_objective(self, x: np.ndarray) -> float:
        return (
            x[0]**2 + x[1]**2 + x[0] * x[1]
            - 14 * x[0] - 16 * x[1]
            + (x[2] - 10)**2 + 4 * (x[3] - 5)**2
            + (x[4] - 3)**2 + 2 * (x[5] - 1)**2
            + 5 * x[6]**2 + 7 * (x[7] - 11)**2
            + 2 * (x[8] - 10)**2 + (x[9] - 7)**2 + 45
        )

    def inequality_constraints(self, x: np.ndarray) -> List[float]:
        g = []
        g.append(-105 + 4 * x[0] + 5 * x[1] - 3 * x[6] + 9 * x[7])
        g.append(10 * x[0] - 8 * x[1] - 17 * x[6] + 2 * x[7])
        g.append(-8 * x[0] + 2 * x[1] + 5 * x[8] - 2 * x[9] - 12)
        g.append(3 * (x[0] - 2)**2 + 4 * (x[1] - 3)**2 + 2 * x[2]**2 - 7 * x[3] - 120)
        g.append(5 * x[0]**2 + 8 * x[1] + (x[2] - 6)**2 - 2 * x[3] - 40)
        g.append(x[0]**2 + 2 * (x[1] - 2)**2 - 2 * x[0] * x[1] + 14 * x[4] - 6 * x[5])
        g.append(0.5 * (x[0] - 8)**2 + 2 * (x[1] - 4)**2 + 3 * x[4]**2 - x[5] - 30)
        g.append(-3 * x[0] + 6 * x[1] + 12 * (x[8] - 8)**2 - 7 * x[9])
        return g

    def _batch_raw_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        return (
            X[:, 0]**2 + X[:, 1]**2 + X[:, 0] * X[:, 1]
            - 14 * X[:, 0] - 16 * X[:, 1]
            + (X[:, 2] - 10)**2 + 4 * (X[:, 3] - 5)**2
            + (X[:, 4] - 3)**2 + 2 * (X[:, 5] - 1)**2
            + 5 * X[:, 6]**2 + 7 * (X[:, 7] - 11)**2
            + 2 * (X[:, 8] - 10)**2 + (X[:, 9] - 7)**2 + 45
        )

    def _batch_inequality_constraints(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n = X.shape[0]
        G = xp.zeros((n, 8))
        G[:, 0] = -105 + 4 * X[:, 0] + 5 * X[:, 1] - 3 * X[:, 6] + 9 * X[:, 7]
        G[:, 1] = 10 * X[:, 0] - 8 * X[:, 1] - 17 * X[:, 6] + 2 * X[:, 7]
        G[:, 2] = -8 * X[:, 0] + 2 * X[:, 1] + 5 * X[:, 8] - 2 * X[:, 9] - 12
        G[:, 3] = 3 * (X[:, 0] - 2)**2 + 4 * (X[:, 1] - 3)**2 + 2 * X[:, 2]**2 - 7 * X[:, 3] - 120
        G[:, 4] = 5 * X[:, 0]**2 + 8 * X[:, 1] + (X[:, 2] - 6)**2 - 2 * X[:, 3] - 40
        G[:, 5] = X[:, 0]**2 + 2 * (X[:, 1] - 2)**2 - 2 * X[:, 0] * X[:, 1] + 14 * X[:, 4] - 6 * X[:, 5]
        G[:, 6] = 0.5 * (X[:, 0] - 8)**2 + 2 * (X[:, 1] - 4)**2 + 3 * X[:, 4]**2 - X[:, 5] - 30
        G[:, 7] = -3 * X[:, 0] + 6 * X[:, 1] + 12 * (X[:, 8] - 8)**2 - 7 * X[:, 9]
        return G


# =============================================================================
# G08: 2-dim, 2 nonlinear inequality constraints
# =============================================================================


class G08(CEC2006Function):
    """G08: Rational objective with 2 nonlinear inequality constraints.

    Dimension: 2
    Constraints: 2 nonlinear inequalities
    Optimal: f* = -0.095825
    """

    _spec = {"func_id": 8, "name": "G08"}
    _n_dim = 2
    _n_linear_ineq = 0
    _n_nonlinear_eq = 0
    _n_nonlinear_ineq = 2
    _f_global = -0.095825
    _x_global = np.array([1.2279713, 4.2453733])
    _variable_bounds = [(0, 10), (0, 10)]

    def raw_objective(self, x: np.ndarray) -> float:
        return -(np.sin(2 * np.pi * x[0])**3 * np.sin(2 * np.pi * x[1])) / (x[0]**3 * (x[0] + x[1]))

    def inequality_constraints(self, x: np.ndarray) -> List[float]:
        g = []
        g.append(x[0]**2 - x[1] + 1)
        g.append(1 - x[0] + (x[1] - 4)**2)
        return g

    def _batch_raw_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        num = xp.sin(2 * np.pi * X[:, 0])**3 * xp.sin(2 * np.pi * X[:, 1])
        denom = X[:, 0]**3 * (X[:, 0] + X[:, 1])
        return -num / denom

    def _batch_inequality_constraints(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n = X.shape[0]
        G = xp.zeros((n, 2))
        G[:, 0] = X[:, 0]**2 - X[:, 1] + 1
        G[:, 1] = 1 - X[:, 0] + (X[:, 1] - 4)**2
        return G


# =============================================================================
# G09: 7-dim, 4 nonlinear inequality constraints
# =============================================================================


class G09(CEC2006Function):
    """G09: Polynomial objective with 4 nonlinear inequality constraints.

    Dimension: 7
    Constraints: 4 nonlinear inequalities
    Optimal: f* = 680.6300573
    """

    _spec = {"func_id": 9, "name": "G09"}
    _n_dim = 7
    _n_linear_ineq = 0
    _n_nonlinear_eq = 0
    _n_nonlinear_ineq = 4
    _f_global = 680.6300573
    _x_global = np.array([
        2.330499, 1.951372, -0.4775414, 4.365726,
        -0.6244870, 1.038131, 1.594227
    ])
    _variable_bounds = [(-10, 10)] * 7

    def raw_objective(self, x: np.ndarray) -> float:
        return (
            (x[0] - 10)**2 + 5 * (x[1] - 12)**2 + x[2]**4
            + 3 * (x[3] - 11)**2 + 10 * x[4]**6 + 7 * x[5]**2
            + x[6]**4 - 4 * x[5] * x[6] - 10 * x[5] - 8 * x[6]
        )

    def inequality_constraints(self, x: np.ndarray) -> List[float]:
        g = []
        g.append(-127 + 2 * x[0]**2 + 3 * x[1]**4 + x[2] + 4 * x[3]**2 + 5 * x[4])
        g.append(-282 + 7 * x[0] + 3 * x[1] + 10 * x[2]**2 + x[3] - x[4])
        g.append(-196 + 23 * x[0] + x[1]**2 + 6 * x[5]**2 - 8 * x[6])
        g.append(4 * x[0]**2 + x[1]**2 - 3 * x[0] * x[1] + 2 * x[2]**2 + 5 * x[5] - 11 * x[6])
        return g

    def _batch_raw_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        return (
            (X[:, 0] - 10)**2 + 5 * (X[:, 1] - 12)**2 + X[:, 2]**4
            + 3 * (X[:, 3] - 11)**2 + 10 * X[:, 4]**6 + 7 * X[:, 5]**2
            + X[:, 6]**4 - 4 * X[:, 5] * X[:, 6] - 10 * X[:, 5] - 8 * X[:, 6]
        )

    def _batch_inequality_constraints(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n = X.shape[0]
        G = xp.zeros((n, 4))
        G[:, 0] = -127 + 2 * X[:, 0]**2 + 3 * X[:, 1]**4 + X[:, 2] + 4 * X[:, 3]**2 + 5 * X[:, 4]
        G[:, 1] = -282 + 7 * X[:, 0] + 3 * X[:, 1] + 10 * X[:, 2]**2 + X[:, 3] - X[:, 4]
        G[:, 2] = -196 + 23 * X[:, 0] + X[:, 1]**2 + 6 * X[:, 5]**2 - 8 * X[:, 6]
        G[:, 3] = 4 * X[:, 0]**2 + X[:, 1]**2 - 3 * X[:, 0] * X[:, 1] + 2 * X[:, 2]**2 + 5 * X[:, 5] - 11 * X[:, 6]
        return G


# =============================================================================
# G10: 8-dim, 3 linear inequality, 3 nonlinear inequality constraints
# =============================================================================


class G10(CEC2006Function):
    """G10: Linear objective with 6 inequality constraints.

    Dimension: 8
    Constraints: 3 linear + 3 nonlinear inequalities
    Optimal: f* = 7049.248
    """

    _spec = {"func_id": 10, "name": "G10"}
    _n_dim = 8
    _n_linear_ineq = 3
    _n_nonlinear_eq = 0
    _n_nonlinear_ineq = 3
    _f_global = 7049.248
    _x_global = np.array([579.3167, 1359.943, 5110.071, 182.0174, 295.5985, 217.9799, 286.4162, 395.5979])
    _variable_bounds = [
        (100, 10000), (1000, 10000), (1000, 10000),
        (10, 1000), (10, 1000), (10, 1000), (10, 1000), (10, 1000)
    ]

    def raw_objective(self, x: np.ndarray) -> float:
        return x[0] + x[1] + x[2]

    def inequality_constraints(self, x: np.ndarray) -> List[float]:
        g = []
        g.append(-1 + 0.0025 * (x[3] + x[5]))
        g.append(-1 + 0.0025 * (x[4] + x[6] - x[3]))
        g.append(-1 + 0.01 * (x[7] - x[4]))
        g.append(-x[0] * x[5] + 833.33252 * x[3] + 100 * x[0] - 83333.333)
        g.append(-x[1] * x[6] + 1250 * x[4] + x[1] * x[3] - 1250 * x[3])
        g.append(-x[2] * x[7] + 1250000 + x[2] * x[4] - 2500 * x[4])
        return g

    def _batch_raw_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        return X[:, 0] + X[:, 1] + X[:, 2]

    def _batch_inequality_constraints(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n = X.shape[0]
        G = xp.zeros((n, 6))
        G[:, 0] = -1 + 0.0025 * (X[:, 3] + X[:, 5])
        G[:, 1] = -1 + 0.0025 * (X[:, 4] + X[:, 6] - X[:, 3])
        G[:, 2] = -1 + 0.01 * (X[:, 7] - X[:, 4])
        G[:, 3] = -X[:, 0] * X[:, 5] + 833.33252 * X[:, 3] + 100 * X[:, 0] - 83333.333
        G[:, 4] = -X[:, 1] * X[:, 6] + 1250 * X[:, 4] + X[:, 1] * X[:, 3] - 1250 * X[:, 3]
        G[:, 5] = -X[:, 2] * X[:, 7] + 1250000 + X[:, 2] * X[:, 4] - 2500 * X[:, 4]
        return G


# =============================================================================
# G11: 2-dim, 1 nonlinear equality constraint
# =============================================================================


class G11(CEC2006Function):
    """G11: Quadratic objective with 1 nonlinear equality constraint.

    Dimension: 2
    Constraints: 1 nonlinear equality
    Optimal: f* = 0.75
    """

    _spec = {"func_id": 11, "name": "G11"}
    _n_dim = 2
    _n_linear_ineq = 0
    _n_nonlinear_eq = 1
    _n_nonlinear_ineq = 0
    _f_global = 0.75
    _x_global = np.array([np.sqrt(0.5), 0.5])
    _variable_bounds = [(-1, 1), (-1, 1)]

    def raw_objective(self, x: np.ndarray) -> float:
        return x[0]**2 + (x[1] - 1)**2

    def equality_constraints(self, x: np.ndarray) -> List[float]:
        return [x[1] - x[0]**2]

    def _batch_raw_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        return X[:, 0]**2 + (X[:, 1] - 1)**2

    def _batch_equality_constraints(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        return (X[:, 1] - X[:, 0]**2).reshape(-1, 1)


# =============================================================================
# G12: 3-dim, 1 nonlinear inequality constraint (disjoint feasible regions)
# =============================================================================


class G12(CEC2006Function):
    """G12: Quadratic objective with disjoint feasible regions.

    Dimension: 3
    Constraints: 1 nonlinear inequality (729 disjoint spherical regions)
    Optimal: f* = -1.0
    """

    _spec = {"func_id": 12, "name": "G12"}
    _n_dim = 3
    _n_linear_ineq = 0
    _n_nonlinear_eq = 0
    _n_nonlinear_ineq = 1
    _f_global = -1.0
    _x_global = np.array([5, 5, 5], dtype=np.float64)
    _variable_bounds = [(0, 10), (0, 10), (0, 10)]

    def raw_objective(self, x: np.ndarray) -> float:
        return -(100 - (x[0] - 5)**2 - (x[1] - 5)**2 - (x[2] - 5)**2) / 100

    def inequality_constraints(self, x: np.ndarray) -> List[float]:
        # g(x) <= 0 is feasible (inside at least one sphere)
        # 729 disjoint spherical regions centered at (p, q, r) for p,q,r in {1,...,9}
        min_dist = float('inf')
        for p in range(1, 10):
            for q in range(1, 10):
                for r in range(1, 10):
                    dist = (x[0] - p)**2 + (x[1] - q)**2 + (x[2] - r)**2
                    min_dist = min(min_dist, dist)
        return [min_dist - 0.0625]  # radius = 0.25, radius^2 = 0.0625

    def _batch_raw_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        return -(100 - (X[:, 0] - 5)**2 - (X[:, 1] - 5)**2 - (X[:, 2] - 5)**2) / 100

    def _batch_inequality_constraints(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n = X.shape[0]
        min_dist = xp.full(n, float('inf'))
        for p in range(1, 10):
            for q in range(1, 10):
                for r in range(1, 10):
                    dist = (X[:, 0] - p)**2 + (X[:, 1] - q)**2 + (X[:, 2] - r)**2
                    min_dist = xp.minimum(min_dist, dist)
        return (min_dist - 0.0625).reshape(-1, 1)


# =============================================================================
# G13: 5-dim, 3 nonlinear equality constraints
# =============================================================================


class G13(CEC2006Function):
    """G13: Exponential objective with 3 nonlinear equality constraints.

    Dimension: 5
    Constraints: 3 nonlinear equalities
    Optimal: f* = 0.053942
    """

    _spec = {"func_id": 13, "name": "G13"}
    _n_dim = 5
    _n_linear_ineq = 0
    _n_nonlinear_eq = 3
    _n_nonlinear_ineq = 0
    _f_global = 0.053942
    _x_global = np.array([-1.717143, 1.595709, 1.827247, -0.7636413, -0.7636450])
    _variable_bounds = [(-2.3, 2.3), (-2.3, 2.3), (-3.2, 3.2), (-3.2, 3.2), (-3.2, 3.2)]

    def raw_objective(self, x: np.ndarray) -> float:
        return np.exp(x[0] * x[1] * x[2] * x[3] * x[4])

    def equality_constraints(self, x: np.ndarray) -> List[float]:
        h = []
        h.append(x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 - 10)
        h.append(x[1] * x[2] - 5 * x[3] * x[4])
        h.append(x[0]**3 + x[1]**3 + 1)
        return h

    def _batch_raw_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        return xp.exp(X[:, 0] * X[:, 1] * X[:, 2] * X[:, 3] * X[:, 4])

    def _batch_equality_constraints(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n = X.shape[0]
        H = xp.zeros((n, 3))
        H[:, 0] = X[:, 0]**2 + X[:, 1]**2 + X[:, 2]**2 + X[:, 3]**2 + X[:, 4]**2 - 10
        H[:, 1] = X[:, 1] * X[:, 2] - 5 * X[:, 3] * X[:, 4]
        H[:, 2] = X[:, 0]**3 + X[:, 1]**3 + 1
        return H


# =============================================================================
# G14: 10-dim, 3 nonlinear equality constraints
# =============================================================================


class G14(CEC2006Function):
    """G14: Logarithmic objective with 3 nonlinear equality constraints.

    Dimension: 10
    Constraints: 3 nonlinear equalities
    Optimal: f* = -47.7649
    """

    _spec = {"func_id": 14, "name": "G14"}
    _n_dim = 10
    _n_linear_ineq = 0
    _n_nonlinear_eq = 3
    _n_nonlinear_ineq = 0
    _f_global = -47.7649
    _x_global = np.array([
        0.0406684, 0.147721, 0.783205, 0.00141433,
        0.485293, 0.000693183, 0.0274052, 0.0179509,
        0.0373268, 0.0968886
    ])
    _variable_bounds = [(0, 10)] * 10

    _c = np.array([-6.089, -17.164, -34.054, -5.914, -24.721,
                   -14.986, -24.100, -10.708, -26.662, -22.179])

    def raw_objective(self, x: np.ndarray) -> float:
        c = self._c
        sum_x = np.sum(x)
        if sum_x <= 0 or np.any(x <= 0):
            return 1e10  # Penalty for invalid domain
        return np.sum(x * (c + np.log(x / sum_x)))

    def equality_constraints(self, x: np.ndarray) -> List[float]:
        h = []
        h.append(x[0] + 2 * x[1] + 2 * x[2] + x[5] + x[9] - 2)
        h.append(x[3] + 2 * x[4] + x[5] + x[6] - 1)
        h.append(x[2] + x[6] + x[7] + 2 * x[8] + x[9] - 1)
        return h

    def _batch_raw_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        c = xp.asarray(self._c)
        sum_x = xp.sum(X, axis=1, keepdims=True)
        # Handle invalid domain
        valid = (sum_x.ravel() > 0) & xp.all(X > 0, axis=1)
        result = xp.full(X.shape[0], 1e10)
        if xp.any(valid):
            log_term = xp.log(X[valid] / sum_x[valid])
            result[valid] = xp.sum(X[valid] * (c + log_term), axis=1)
        return result

    def _batch_equality_constraints(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n = X.shape[0]
        H = xp.zeros((n, 3))
        H[:, 0] = X[:, 0] + 2 * X[:, 1] + 2 * X[:, 2] + X[:, 5] + X[:, 9] - 2
        H[:, 1] = X[:, 3] + 2 * X[:, 4] + X[:, 5] + X[:, 6] - 1
        H[:, 2] = X[:, 2] + X[:, 6] + X[:, 7] + 2 * X[:, 8] + X[:, 9] - 1
        return H


# =============================================================================
# G15: 3-dim, 2 nonlinear equality constraints
# =============================================================================


class G15(CEC2006Function):
    """G15: Polynomial objective with 2 nonlinear equality constraints.

    Dimension: 3
    Constraints: 2 nonlinear equalities
    Optimal: f* = 961.715
    """

    _spec = {"func_id": 15, "name": "G15"}
    _n_dim = 3
    _n_linear_ineq = 0
    _n_nonlinear_eq = 2
    _n_nonlinear_ineq = 0
    _f_global = 961.715
    _x_global = np.array([3.51212812611795133, 0.216987510429556135, 3.55217854929179921])
    _variable_bounds = [(0, 10), (0, 10), (0, 10)]

    def raw_objective(self, x: np.ndarray) -> float:
        return 1000 - x[0]**2 - 2 * x[1]**2 - x[2]**2 - x[0] * x[1] - x[0] * x[2]

    def equality_constraints(self, x: np.ndarray) -> List[float]:
        h = []
        h.append(x[0]**2 + x[1]**2 + x[2]**2 - 25)
        h.append(8 * x[0] + 14 * x[1] + 7 * x[2] - 56)
        return h

    def _batch_raw_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        return 1000 - X[:, 0]**2 - 2 * X[:, 1]**2 - X[:, 2]**2 - X[:, 0] * X[:, 1] - X[:, 0] * X[:, 2]

    def _batch_equality_constraints(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n = X.shape[0]
        H = xp.zeros((n, 2))
        H[:, 0] = X[:, 0]**2 + X[:, 1]**2 + X[:, 2]**2 - 25
        H[:, 1] = 8 * X[:, 0] + 14 * X[:, 1] + 7 * X[:, 2] - 56
        return H


# =============================================================================
# G16: 5-dim, 4 linear inequality, 34 nonlinear inequality constraints
# =============================================================================


class G16(CEC2006Function):
    """G16: Complex industrial process optimization.

    Dimension: 5
    Constraints: 38 nonlinear inequalities (simplified from mixed linear/nonlinear)
    Optimal: f* = -1.9052
    """

    _spec = {"func_id": 16, "name": "G16"}
    _n_dim = 5
    _n_linear_ineq = 0
    _n_nonlinear_eq = 0
    _n_nonlinear_ineq = 34  # Simplified constraint count
    _f_global = -1.9052
    _x_global = np.array([705.1803, 68.60, 102.9, 282.3, 37.58])
    _variable_bounds = [(704.4148, 906.3855), (68.6, 288.88), (0, 134.75),
                        (193, 287.0966), (25, 84.1988)]

    def _compute_y(self, x: np.ndarray):
        """Compute intermediate variables y1-y17."""
        y = np.zeros(18)  # y[1] to y[17]
        c = np.zeros(18)
        c[1], c[2], c[3], c[4] = 0.0, 0.0, 0.0, 0.0

        y[1] = x[1] + x[2] + 41.6
        c[1] = 0.024 * x[3] - 4.62
        y[2] = 12.5 / c[1] + 12
        c[2] = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y[2] * x[0]
        c[3] = 0.052 * x[0] + 78 + 0.002377 * y[2] * x[0]
        y[3] = c[2] / c[3]
        y[4] = 19 * y[3]
        c[4] = 0.04782 * (x[0] - y[3]) + (0.1956 * (x[0] - y[3])**2) / x[1] + 0.6376 * y[4] + 1.594 * y[3]
        c[5] = 100 * x[1]
        c[6] = x[0] - y[3] - y[4]
        c[7] = 0.950 - c[4] / c[5]
        y[5] = c[6] * c[7]
        y[6] = x[0] - y[5] - y[4] - y[3]
        c[8] = (y[5] + y[4]) * 0.995
        y[7] = c[8] / y[1]
        y[8] = c[8] / 3798
        c[9] = y[6] * 0.0163 / 798
        y[9] = c[9] * x[3]
        y[10] = 96.82 / c[9] + 0.321 * y[1]
        y[11] = 1.29 * y[5] + 1.258 * y[4] + 2.29 * y[3] + 1.71 * y[6]
        y[12] = 1.71 * x[0] - 0.452 * y[4] + 0.580 * y[3]
        c[10] = 12.3 / 752.3
        c[11] = 1.75 * y[1] * 0.995 * x[0]
        c[12] = 0.995 * y[9] + 1998
        y[13] = c[10] * x[0] + c[11] / c[12]
        y[14] = c[12] - 1.75 * y[1]
        y[15] = y[13] + 1.5 * x[1]
        y[16] = 0.7302 * y[15] * (y[14] * 0.006 + 0.004433) + 0.0588
        c[13] = y[16] / (0.5 * x[3])
        c[14] = 0.0144 * x[4] + 1
        y[17] = (c[13] - 0.0039 * x[4]) / c[14]
        return y

    def raw_objective(self, x: np.ndarray) -> float:
        y = self._compute_y(x)
        return (
            0.000117 * y[14] + 0.1365
            + 0.00002358 * y[13] + 0.000001502 * y[16]
            + 0.0321 * y[12] + 0.004324 * y[5] + 0.0001 * (y[17] / y[15]) + 37.48 * (y[1] / y[12])
            - 0.0000005843 * y[17]
        )

    def inequality_constraints(self, x: np.ndarray) -> List[float]:
        y = self._compute_y(x)
        g = []
        # Linear constraints (approximated from bounds)
        g.append(0.28 / 0.72 * y[5] - y[4])
        g.append(x[2] - 1.5 * x[1])
        g.append(3496 * y[1] / y[12] - 21)
        g.append(110.6 + y[1] - 62212 / y[12])
        # Nonlinear constraints from function definition
        g.append(y[3] - 0.28 / 0.72 * y[5])
        g.append(1.104 - 0.72 * y[5])
        g.append(y[9] - 15000)
        g.append(y[9] * 0.9 - y[7])
        g.append(y[7] - y[9] * 0.7)
        g.append(y[10] - y[8] * 0.64)
        g.append(y[8] * 0.64 - y[10])
        g.append(y[11] - 4)
        g.append(1 / y[11] - 0.25)
        g.append(y[12] + 18)
        g.append(y[5] - y[17] + y[4])
        g.append(0.32 - y[9] * 0.00061)
        g.append(y[9] * 0.00061 - 0.32)
        g.append(y[17] - 0.574 * y[5])
        g.append(y[9] - 4)
        g.append(4 - y[9])
        g.append(y[9] - 9.9)
        g.append(y[6] * 9.8 / x[1])
        g.append(-y[6] * 9.8 / x[1])
        g.append(y[1] - 61)
        g.append(61 - y[1])
        g.append(y[2] - 21)
        g.append(21 - y[2])
        g.append(y[3] - 0.9 * y[5])
        g.append(0.1 * y[5] - y[3])
        g.append(y[4] - 0.35 * y[5])
        g.append(y[4] - 0.1 * y[5])
        g.append(y[5] - 0.85 * y[6])
        g.append(0.85 * y[6] - y[5])
        g.append(y[8] - y[7])
        return g

    def _batch_raw_objective(self, X: ArrayLike) -> ArrayLike:
        # Fall back to sequential for complex function
        xp = get_array_namespace(X)
        results = xp.zeros(X.shape[0])
        for i in range(X.shape[0]):
            results[i] = self.raw_objective(np.asarray(X[i]))
        return results

    def _batch_inequality_constraints(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n = X.shape[0]
        n_ineq = self._n_linear_ineq + self._n_nonlinear_ineq
        G = xp.zeros((n, n_ineq))
        for i in range(n):
            constraints = self.inequality_constraints(np.asarray(X[i]))
            G[i, :len(constraints)] = xp.asarray(constraints)
        return G


# =============================================================================
# G17: 6-dim, 4 nonlinear equality constraints
# =============================================================================


class G17(CEC2006Function):
    """G17: Piecewise function with 4 nonlinear equality constraints.

    Dimension: 6
    Constraints: 4 nonlinear equalities
    Optimal: f* = 8853.5397
    """

    _spec = {"func_id": 17, "name": "G17"}
    _n_dim = 6
    _n_linear_ineq = 0
    _n_nonlinear_eq = 4
    _n_nonlinear_ineq = 0
    _f_global = 8853.5397
    _x_global = np.array([201.784467214523659, 99.9999999999999005, 383.071034852773266,
                          420.0, -10.9076584514292652, 0.0731482312084287128])
    _variable_bounds = [(0, 400), (0, 1000), (340, 420), (340, 420), (-1000, 1000), (0, 0.5236)]

    def raw_objective(self, x: np.ndarray) -> float:
        f1 = 30 * x[0] if x[0] < 300 else 31 * x[0]
        f2 = 28 * x[1] if x[1] < 100 else 29 * x[1] if x[1] < 200 else 30 * x[1]
        return f1 + f2

    def equality_constraints(self, x: np.ndarray) -> List[float]:
        h = []
        h.append(-x[0] + 300 - x[2] * x[3] / 131.078 * np.cos(1.48477 - x[5])
                 + 0.90798 * x[2]**2 / 131.078 * np.cos(1.47588))
        h.append(-x[1] - x[2] * x[3] / 131.078 * np.cos(1.48477 + x[5])
                 + 0.90798 * x[3]**2 / 131.078 * np.cos(1.47588))
        h.append(-x[4] - x[2] * x[3] / 131.078 * np.sin(1.48477 + x[5])
                 + 0.90798 * x[3]**2 / 131.078 * np.sin(1.47588))
        h.append(200 - x[2] * x[3] / 131.078 * np.sin(1.48477 - x[5])
                 + 0.90798 * x[2]**2 / 131.078 * np.sin(1.47588))
        return h

    def _batch_raw_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        f1 = xp.where(X[:, 0] < 300, 30 * X[:, 0], 31 * X[:, 0])
        f2 = xp.where(X[:, 1] < 100, 28 * X[:, 1],
                      xp.where(X[:, 1] < 200, 29 * X[:, 1], 30 * X[:, 1]))
        return f1 + f2

    def _batch_equality_constraints(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n = X.shape[0]
        H = xp.zeros((n, 4))
        H[:, 0] = (-X[:, 0] + 300 - X[:, 2] * X[:, 3] / 131.078 * xp.cos(1.48477 - X[:, 5])
                   + 0.90798 * X[:, 2]**2 / 131.078 * xp.cos(1.47588))
        H[:, 1] = (-X[:, 1] - X[:, 2] * X[:, 3] / 131.078 * xp.cos(1.48477 + X[:, 5])
                   + 0.90798 * X[:, 3]**2 / 131.078 * xp.cos(1.47588))
        H[:, 2] = (-X[:, 4] - X[:, 2] * X[:, 3] / 131.078 * xp.sin(1.48477 + X[:, 5])
                   + 0.90798 * X[:, 3]**2 / 131.078 * xp.sin(1.47588))
        H[:, 3] = (200 - X[:, 2] * X[:, 3] / 131.078 * xp.sin(1.48477 - X[:, 5])
                   + 0.90798 * X[:, 2]**2 / 131.078 * xp.sin(1.47588))
        return H


# =============================================================================
# G18: 9-dim, 13 nonlinear inequality constraints
# =============================================================================


class G18(CEC2006Function):
    """G18: Quadratic objective with 13 nonlinear inequality constraints.

    Dimension: 9
    Constraints: 13 nonlinear inequalities
    Optimal: f* = -0.866025
    """

    _spec = {"func_id": 18, "name": "G18"}
    _n_dim = 9
    _n_linear_ineq = 0
    _n_nonlinear_eq = 0
    _n_nonlinear_ineq = 13
    _f_global = -0.866025
    _x_global = np.array([
        -0.657776192427943163, -0.153418773482438542, 0.323413871675240938,
        -0.946257611651304398, -0.657776194376798906, -0.753213434632691414,
        0.323413874123576972, -0.346462947962331735, 0.59979466285217542
    ])
    _variable_bounds = [(-10, 10)] * 8 + [(0, 20)]

    def raw_objective(self, x: np.ndarray) -> float:
        return -0.5 * (x[0] * x[3] - x[1] * x[2] + x[2] * x[8]
                       - x[4] * x[8] + x[4] * x[7] - x[5] * x[6])

    def inequality_constraints(self, x: np.ndarray) -> List[float]:
        g = []
        g.append(x[2]**2 + x[3]**2 - 1)
        g.append(x[8]**2 - 1)
        g.append(x[4]**2 + x[5]**2 - 1)
        g.append(x[0]**2 + (x[1] - x[8])**2 - 1)
        g.append((x[0] - x[4])**2 + (x[1] - x[5])**2 - 1)
        g.append((x[0] - x[6])**2 + (x[1] - x[7])**2 - 1)
        g.append((x[2] - x[4])**2 + (x[3] - x[5])**2 - 1)
        g.append((x[2] - x[6])**2 + (x[3] - x[7])**2 - 1)
        g.append(x[6]**2 + (x[7] - x[8])**2 - 1)
        g.append(x[1] * x[2] - x[0] * x[3])
        g.append(-x[2] * x[8])
        g.append(x[4] * x[8])
        g.append(x[5] * x[6] - x[4] * x[7])
        return g

    def _batch_raw_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        return -0.5 * (X[:, 0] * X[:, 3] - X[:, 1] * X[:, 2] + X[:, 2] * X[:, 8]
                       - X[:, 4] * X[:, 8] + X[:, 4] * X[:, 7] - X[:, 5] * X[:, 6])

    def _batch_inequality_constraints(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n = X.shape[0]
        G = xp.zeros((n, 13))
        G[:, 0] = X[:, 2]**2 + X[:, 3]**2 - 1
        G[:, 1] = X[:, 8]**2 - 1
        G[:, 2] = X[:, 4]**2 + X[:, 5]**2 - 1
        G[:, 3] = X[:, 0]**2 + (X[:, 1] - X[:, 8])**2 - 1
        G[:, 4] = (X[:, 0] - X[:, 4])**2 + (X[:, 1] - X[:, 5])**2 - 1
        G[:, 5] = (X[:, 0] - X[:, 6])**2 + (X[:, 1] - X[:, 7])**2 - 1
        G[:, 6] = (X[:, 2] - X[:, 4])**2 + (X[:, 3] - X[:, 5])**2 - 1
        G[:, 7] = (X[:, 2] - X[:, 6])**2 + (X[:, 3] - X[:, 7])**2 - 1
        G[:, 8] = X[:, 6]**2 + (X[:, 7] - X[:, 8])**2 - 1
        G[:, 9] = X[:, 1] * X[:, 2] - X[:, 0] * X[:, 3]
        G[:, 10] = -X[:, 2] * X[:, 8]
        G[:, 11] = X[:, 4] * X[:, 8]
        G[:, 12] = X[:, 5] * X[:, 6] - X[:, 4] * X[:, 7]
        return G


# =============================================================================
# G19: 15-dim, 5 nonlinear inequality constraints
# =============================================================================


class G19(CEC2006Function):
    """G19: Quadratic objective with 5 nonlinear inequality constraints.

    Dimension: 15
    Constraints: 5 nonlinear inequalities
    Optimal: f* = 32.6556
    """

    _spec = {"func_id": 19, "name": "G19"}
    _n_dim = 15
    _n_linear_ineq = 0
    _n_nonlinear_eq = 0
    _n_nonlinear_ineq = 5
    _f_global = 32.6556
    _x_global = None  # Complex optimal solution
    _variable_bounds = [(0, 10)] * 15

    # Problem matrices
    _a = np.array([
        [-16, 2, 0, 1, 0],
        [0, -2, 0, 0.4, 2],
        [-3.5, 0, 2, 0, 0],
        [0, -2, 0, -4, -1],
        [0, -9, -2, 1, -2.8],
        [2, 0, -4, 0, 0],
        [-1, -1, -1, -1, -1],
        [-1, -2, -3, -2, -1],
        [1, 2, 3, 4, 5],
        [1, 1, 1, 1, 1]
    ])
    _b = np.array([-40, -2, -0.25, -4, -4, -1, -40, -60, 5, 1])
    _c = np.array([
        [30, -20, -10, 32, -10],
        [-20, 39, -6, -31, 32],
        [-10, -6, 10, -6, -10],
        [32, -31, -6, 39, -20],
        [-10, 32, -10, -20, 30]
    ])
    _d = np.array([4, 8, 10, 6, 2])
    _e = np.array([-15, -27, -36, -18, -12])

    def raw_objective(self, x: np.ndarray) -> float:
        a, b, c, d, e = self._a, self._b, self._c, self._d, self._e
        sum1 = np.sum(c * np.outer(x[10:15], x[10:15]))
        sum2 = 2 * np.sum(d * x[10:15]**3)
        sum3 = np.sum(b * x[:10])
        return sum1 + sum2 - sum3

    def inequality_constraints(self, x: np.ndarray) -> List[float]:
        a, b, c, d, e = self._a, self._b, self._c, self._d, self._e
        g = []
        for j in range(5):
            val = -2 * np.sum(c[j, :] * x[10:15]) - 3 * d[j] * x[10 + j]**2
            val -= e[j] + np.sum(a[:, j] * x[:10])
            g.append(val)
        return g

    def _batch_raw_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        c, d, b = xp.asarray(self._c), xp.asarray(self._d), xp.asarray(self._b)
        n = X.shape[0]
        result = xp.zeros(n)
        for i in range(n):
            x = X[i]
            sum1 = xp.sum(c * xp.outer(x[10:15], x[10:15]))
            sum2 = 2 * xp.sum(d * x[10:15]**3)
            sum3 = xp.sum(b * x[:10])
            result[i] = sum1 + sum2 - sum3
        return result

    def _batch_inequality_constraints(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        a, c, d, e = xp.asarray(self._a), xp.asarray(self._c), xp.asarray(self._d), xp.asarray(self._e)
        n = X.shape[0]
        G = xp.zeros((n, 5))
        for i in range(n):
            x = X[i]
            for j in range(5):
                val = -2 * xp.sum(c[j, :] * x[10:15]) - 3 * d[j] * x[10 + j]**2
                val -= e[j] + xp.sum(a[:, j] * x[:10])
                G[i, j] = val
        return G


# =============================================================================
# G20: 24-dim, 6 nonlinear inequality, 20 nonlinear equality constraints
# =============================================================================


class G20(CEC2006Function):
    """G20: Complex chemical process (no known feasible solution).

    Dimension: 24
    Constraints: 6 nonlinear inequalities, 20 nonlinear equalities
    Optimal: Unknown (problem may be infeasible)
    """

    _spec = {"func_id": 20, "name": "G20"}
    _n_dim = 24
    _n_linear_ineq = 0
    _n_nonlinear_eq = 20
    _n_nonlinear_ineq = 6
    _f_global = 0.0  # Unknown
    _x_global = None
    _variable_bounds = [(0, 10)] * 24

    # Problem constants
    _a = np.array([0.0693, 0.0577, 0.05, 0.2, 0.26, 0.55, 0.06, 0.1, 0.12, 0.18,
                   0.1, 0.09, 0.0693, 0.0577, 0.05, 0.2, 0.26, 0.55, 0.06, 0.1, 0.12, 0.18, 0.1, 0.09])
    _b = np.array([44.094, 58.12, 58.12, 137.4, 120.9, 170.9, 62.501, 84.94, 133.425, 82.507,
                   46.07, 60.097, 44.094, 58.12, 58.12, 137.4, 120.9, 170.9, 62.501, 84.94, 133.425, 82.507,
                   46.07, 60.097])
    _c = np.array([123.7, 31.7, 45.7, 14.7, 84.7, 27.7, 49.7, 7.1, 2.1, 17.7, 0.85, 0.64])
    _d = np.array([31.244, 36.12, 34.784, 92.7, 82.7, 91.6, 56.708, 82.7, 80.8, 64.517, 49.4, 49.1])
    _e = np.array([0.1, 0.3, 0.4, 0.3, 0.6, 0.3])

    def raw_objective(self, x: np.ndarray) -> float:
        return np.sum(self._a * x)

    def inequality_constraints(self, x: np.ndarray) -> List[float]:
        g = []
        for j in range(6):
            val = x[j] + x[j + 12] - self._e[j]
            g.append(val)
        return g

    def equality_constraints(self, x: np.ndarray) -> List[float]:
        h = []
        # Mass balance constraints
        sum_x = np.sum(x[:12])
        sum_x2 = np.sum(x[12:24])
        for k in range(12):
            h.append(x[k] / (self._b[k] * sum_x) - x[k + 12] / (self._b[k + 12] * sum_x2))
        # Additional constraints
        for k in range(4):
            h.append(x[k] + x[k + 12] - self._c[k])
        for k in range(4):
            h.append(x[k + 4] + x[k + 16] - self._d[k])
        return h

    def _batch_raw_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        a = xp.asarray(self._a)
        return xp.sum(a * X, axis=1)

    def _batch_inequality_constraints(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        e = xp.asarray(self._e)
        n = X.shape[0]
        G = xp.zeros((n, 6))
        for j in range(6):
            G[:, j] = X[:, j] + X[:, j + 12] - e[j]
        return G

    def _batch_equality_constraints(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        b, c, d = xp.asarray(self._b), xp.asarray(self._c), xp.asarray(self._d)
        n = X.shape[0]
        H = xp.zeros((n, 20))
        sum_x = xp.sum(X[:, :12], axis=1, keepdims=True)
        sum_x2 = xp.sum(X[:, 12:24], axis=1, keepdims=True)
        for k in range(12):
            H[:, k] = X[:, k] / (b[k] * sum_x.ravel()) - X[:, k + 12] / (b[k + 12] * sum_x2.ravel())
        for k in range(4):
            H[:, 12 + k] = X[:, k] + X[:, k + 12] - c[k]
        for k in range(4):
            H[:, 16 + k] = X[:, k + 4] + X[:, k + 16] - d[k]
        return H


# =============================================================================
# G21: 7-dim, 1 nonlinear inequality, 5 nonlinear equality constraints
# =============================================================================


class G21(CEC2006Function):
    """G21: Quadratic objective with 6 constraints.

    Dimension: 7
    Constraints: 1 nonlinear inequality, 5 nonlinear equalities
    Optimal: f* = 193.7245
    """

    _spec = {"func_id": 21, "name": "G21"}
    _n_dim = 7
    _n_linear_ineq = 0
    _n_nonlinear_eq = 5
    _n_nonlinear_ineq = 1
    _f_global = 193.7245
    _x_global = np.array([
        193.724510070034967, 5.56944131553368433e-27, 17.3191887294084914,
        100.047897801386839, 6.68445185362377892, 5.99168428444264833, 6.21451648886070451
    ])
    _variable_bounds = [(0, 1000), (0, 40), (0, 40), (100, 300), (6.3, 6.7), (5.9, 6.4), (4.5, 6.25)]

    def raw_objective(self, x: np.ndarray) -> float:
        return x[0]

    def inequality_constraints(self, x: np.ndarray) -> List[float]:
        return [-x[0] + 35 * x[1]**0.6 + 35 * x[2]**0.6]

    def equality_constraints(self, x: np.ndarray) -> List[float]:
        h = []
        h.append(-300 * x[2] + 7500 * x[4] - 7500 * x[5] - 25 * x[3] * x[4] + 25 * x[3] * x[5] + x[2] * x[3])
        h.append(100 * x[1] + 155.365 * x[3] + 2500 * x[6] - x[1] * x[3] - 25 * x[3] * x[6] - 15536.5)
        h.append(-x[4] + np.log(-x[3] + 900))
        h.append(-x[5] + np.log(x[3] + 300))
        h.append(-x[6] + np.log(-2 * x[3] + 700))
        return h

    def _batch_raw_objective(self, X: ArrayLike) -> ArrayLike:
        return X[:, 0]

    def _batch_inequality_constraints(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        return (-X[:, 0] + 35 * X[:, 1]**0.6 + 35 * X[:, 2]**0.6).reshape(-1, 1)

    def _batch_equality_constraints(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n = X.shape[0]
        H = xp.zeros((n, 5))
        H[:, 0] = -300 * X[:, 2] + 7500 * X[:, 4] - 7500 * X[:, 5] - 25 * X[:, 3] * X[:, 4] + 25 * X[:, 3] * X[:, 5] + X[:, 2] * X[:, 3]
        H[:, 1] = 100 * X[:, 1] + 155.365 * X[:, 3] + 2500 * X[:, 6] - X[:, 1] * X[:, 3] - 25 * X[:, 3] * X[:, 6] - 15536.5
        H[:, 2] = -X[:, 4] + xp.log(xp.maximum(-X[:, 3] + 900, 1e-10))
        H[:, 3] = -X[:, 5] + xp.log(xp.maximum(X[:, 3] + 300, 1e-10))
        H[:, 4] = -X[:, 6] + xp.log(xp.maximum(-2 * X[:, 3] + 700, 1e-10))
        return H


# =============================================================================
# G22: 22-dim, 1 nonlinear inequality, 19 nonlinear equality constraints
# =============================================================================


class G22(CEC2006Function):
    """G22: Linear objective with 20 constraints.

    Dimension: 22
    Constraints: 1 nonlinear inequality, 19 nonlinear equalities
    Optimal: f* = 236.4309 (approximate)
    """

    _spec = {"func_id": 22, "name": "G22"}
    _n_dim = 22
    _n_linear_ineq = 0
    _n_nonlinear_eq = 19
    _n_nonlinear_ineq = 1
    _f_global = 236.4309
    _x_global = None  # Complex optimal
    _variable_bounds = [
        (0, 20000), (0, 1e6), (0, 1e6), (0, 1e6), (0, 4e7),
        (100, 4e7), (100, 4e7), (100, 4e7), (100, 4e7), (100, 4e7),
        (100, 2e4), (100, 2e4), (0, 1e7), (0, 1e7), (0, 1e7),
        (0.01, 5e4), (0, 5e4), (0, 5e4), (0, 5e4), (0, 5e4),
        (0, 200), (0, 16)
    ]

    def raw_objective(self, x: np.ndarray) -> float:
        return x[0]

    def inequality_constraints(self, x: np.ndarray) -> List[float]:
        return [-x[0] + x[1]**0.6 + x[2]**0.6 + x[3]**0.6]

    def equality_constraints(self, x: np.ndarray) -> List[float]:
        h = []
        h.append(x[4] - 100000 * x[7] + 1e7)
        h.append(x[5] + 100000 * x[7] - 100000 * x[8])
        h.append(x[6] + 100000 * x[8] - 5e7)
        h.append(x[4] - 100000 * x[10] + 3.3e7)
        h.append(x[5] + 100000 * x[10] - 100000 * x[11] - 4.4e7)
        h.append(x[6] + 100000 * x[11] - 6.6e7)
        h.append(x[4] - 120 * x[1] * x[12])
        h.append(x[5] - 80 * x[2] * x[13])
        h.append(x[6] - 40 * x[3] * x[14])
        h.append(x[7] - x[10] + x[15])
        h.append(x[8] - x[11] + x[16])
        h.append(-x[17] + np.log(x[9] - 100))
        h.append(-x[18] + np.log(-x[7] + 300))
        h.append(-x[19] + np.log(x[15]))
        h.append(-x[20] + np.log(-x[8] + 400))
        h.append(-x[21] + np.log(x[16]))
        h.append(-x[7] - x[9] + x[12] * x[17] - x[12] * x[18] + 400)
        h.append(x[7] - x[8] - x[10] + x[13] * x[18] - x[13] * x[19] + 400)
        h.append(x[8] - x[11] - 4.6e2 + x[14] * x[19])
        return h

    def _batch_raw_objective(self, X: ArrayLike) -> ArrayLike:
        return X[:, 0]

    def _batch_inequality_constraints(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        return (-X[:, 0] + X[:, 1]**0.6 + X[:, 2]**0.6 + X[:, 3]**0.6).reshape(-1, 1)

    def _batch_equality_constraints(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n = X.shape[0]
        H = xp.zeros((n, 19))
        # Complex equality constraints - fall back to sequential
        for i in range(n):
            x = np.asarray(X[i])
            constraints = self.equality_constraints(x)
            H[i, :] = xp.asarray(constraints)
        return H


# =============================================================================
# G23: 9-dim, 2 nonlinear inequality, 4 nonlinear equality constraints
# =============================================================================


class G23(CEC2006Function):
    """G23: Linear objective with 6 constraints.

    Dimension: 9
    Constraints: 2 nonlinear inequalities, 4 nonlinear equalities
    Optimal: f* = -400.055 (approximate, no closed-form optimal known)
    """

    _spec = {"func_id": 23, "name": "G23"}
    _n_dim = 9
    _n_linear_ineq = 0
    _n_nonlinear_eq = 4
    _n_nonlinear_ineq = 2
    _f_global = -400.055
    _x_global = None  # No closed-form optimal solution known
    _variable_bounds = [(0, 300), (0, 300), (0, 100), (0, 200), (0, 100),
                        (0, 300), (0, 100), (0, 200), (0.01, 0.03)]

    def raw_objective(self, x: np.ndarray) -> float:
        return -9 * x[4] - 15 * x[7] + 6 * x[0] + 16 * x[1] + 10 * (x[5] + x[6])

    def inequality_constraints(self, x: np.ndarray) -> List[float]:
        g = []
        g.append(x[8] * x[2] + 0.02 * x[5] - 0.025 * x[4])
        g.append(x[8] * x[3] + 0.02 * x[6] - 0.015 * x[7])
        return g

    def equality_constraints(self, x: np.ndarray) -> List[float]:
        h = []
        h.append(x[0] + x[1] - x[2] - x[3])
        h.append(0.03 * x[0] + 0.01 * x[1] - x[8] * (x[2] + x[3]))
        h.append(x[2] + x[5] - x[4])
        h.append(x[3] + x[6] - x[7])
        return h

    def _batch_raw_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        return -9 * X[:, 4] - 15 * X[:, 7] + 6 * X[:, 0] + 16 * X[:, 1] + 10 * (X[:, 5] + X[:, 6])

    def _batch_inequality_constraints(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n = X.shape[0]
        G = xp.zeros((n, 2))
        G[:, 0] = X[:, 8] * X[:, 2] + 0.02 * X[:, 5] - 0.025 * X[:, 4]
        G[:, 1] = X[:, 8] * X[:, 3] + 0.02 * X[:, 6] - 0.015 * X[:, 7]
        return G

    def _batch_equality_constraints(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n = X.shape[0]
        H = xp.zeros((n, 4))
        H[:, 0] = X[:, 0] + X[:, 1] - X[:, 2] - X[:, 3]
        H[:, 1] = 0.03 * X[:, 0] + 0.01 * X[:, 1] - X[:, 8] * (X[:, 2] + X[:, 3])
        H[:, 2] = X[:, 2] + X[:, 5] - X[:, 4]
        H[:, 3] = X[:, 3] + X[:, 6] - X[:, 7]
        return H


# =============================================================================
# G24: 2-dim, 2 nonlinear inequality constraints
# =============================================================================


class G24(CEC2006Function):
    """G24: Quadratic objective with 2 nonlinear inequality constraints.

    Dimension: 2
    Constraints: 2 nonlinear inequalities
    Optimal: f* = -5.5080
    """

    _spec = {"func_id": 24, "name": "G24"}
    _n_dim = 2
    _n_linear_ineq = 0
    _n_nonlinear_eq = 0
    _n_nonlinear_ineq = 2
    _f_global = -5.5080
    _x_global = np.array([2.32952, 3.17849])
    _variable_bounds = [(0, 3), (0, 4)]

    def raw_objective(self, x: np.ndarray) -> float:
        return -x[0] - x[1]

    def inequality_constraints(self, x: np.ndarray) -> List[float]:
        g = []
        g.append(-2 * x[0]**4 + 8 * x[0]**3 - 8 * x[0]**2 + x[1] - 2)
        g.append(-4 * x[0]**4 + 32 * x[0]**3 - 88 * x[0]**2 + 96 * x[0] + x[1] - 36)
        return g

    def _batch_raw_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        return -X[:, 0] - X[:, 1]

    def _batch_inequality_constraints(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n = X.shape[0]
        G = xp.zeros((n, 2))
        G[:, 0] = -2 * X[:, 0]**4 + 8 * X[:, 0]**3 - 8 * X[:, 0]**2 + X[:, 1] - 2
        G[:, 1] = -4 * X[:, 0]**4 + 32 * X[:, 0]**3 - 88 * X[:, 0]**2 + 96 * X[:, 0] + X[:, 1] - 36
        return G


# =============================================================================
# All CEC 2006 functions
# =============================================================================

CEC2006_ALL = [
    G01, G02, G03, G04, G05, G06, G07, G08, G09, G10,
    G11, G12, G13, G14, G15, G16, G17, G18, G19, G20,
    G21, G22, G23, G24
]
