# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from sympy import Sum, Indexed
from sympy.abc import x, i

from .base_objective_function import ObjectiveFunction


class SphereFunction(ObjectiveFunction):
    __name__ = "sphere_function"

    def __init__(
        self, n_dim, alpha=1, omega=0, metric="score", input_type="dictionary"
    ):
        super().__init__(metric, input_type)
        self.latex = r"f(x)=\sum_{i=1}^{" + str(n_dim) + "} x_i^{2}"
        self.sympy = Sum(Indexed(x, i) ** 2, (i, 1, n_dim))

        self.n_dim = n_dim

        self.alpha = alpha
        self.omega = omega

    def objective_function_dict(self, params):
        loss = 0
        for dim in range(self.n_dim):
            dim_str = "x" + str(dim)
            x = params[dim_str]

            loss += x * x

        loss = self.alpha * loss + self.omega

        return self.return_metric(loss)


class AckleyFunction(ObjectiveFunction):
    __name__ = "ackley_function"

    def __init__(
        self,
        alpha=1,
        beta=1,
        gamma=1,
        delta=1,
        omega=0,
        A=20,
        angle=2 * np.pi,
        metric="score",
        input_type="dictionary",
    ):
        super().__init__(metric, input_type)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.omega = omega

        self.A = A
        self.angle = angle

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss1 = -self.A * np.exp(-0.2 * np.sqrt(0.5 * (x * x + y * y)))
        loss2 = -np.exp(0.5 * (np.cos(self.angle * x) + np.cos(self.angle * y)))
        loss3 = np.exp(1)
        loss4 = self.A

        loss = (
            self.alpha * loss1
            + self.beta * loss2
            + self.gamma * loss3
            + self.delta * loss4
            + self.omega
        )

        return self.return_metric(loss)


class RastriginFunction(ObjectiveFunction):
    __name__ = "rastrigin_function"

    def __init__(
        self,
        n_dim,
        alpha=1,
        beta=1,
        omega=0,
        A=10,
        angle=2 * np.pi,
        metric="score",
        input_type="dictionary",
    ):
        super().__init__(metric, input_type)

        self.n_dim = n_dim

        self.alpha = alpha
        self.beta = beta
        self.omega = omega

        self.A = A
        self.angle = angle

    def objective_function_dict(self, params):
        loss1 = self.A * self.n_dim

        loss2 = 0
        for dim in range(self.n_dim):
            dim_str = "x" + str(dim)
            x = params[dim_str]

            loss2 += x * x - self.A * np.cos(self.angle * x)

        loss = self.alpha * loss1 + self.beta * loss2 + self.omega

        return self.return_metric(loss)


class RosenbrockFunction(ObjectiveFunction):
    __name__ = "rosenbrock_function"

    def __init__(
        self,
        n_dim,
        alpha=1,
        omega=0,
        A=1,
        B=100,
        metric="score",
        input_type="dictionary",
    ):
        super().__init__(metric, input_type)

        self.n_dim = n_dim

        self.alpha = alpha
        self.omega = omega

        self.A = A
        self.B = B

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss = 0
        for dim in range(self.n_dim):
            dim_str = "x" + str(dim)
            x = params[dim_str]

            loss = (self.A - x) ** 2 + self.B * (y - x ** 2) ** 2

        loss = self.alpha * loss + self.omega

        return self.return_metric(loss)


class BealeFunction(ObjectiveFunction):
    __name__ = "beale_function"

    def __init__(
        self,
        alpha=1,
        beta=1,
        gamma=1,
        omega=0,
        A=1.5,
        B=2.25,
        C=2.652,
        metric="score",
        input_type="dictionary",
    ):
        super().__init__(metric, input_type)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.omega = omega

        self.A = A
        self.B = B
        self.C = C

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss1 = (self.A - x + x * y) ** 2
        loss2 = (self.B - x + x * y ** 2) ** 2
        loss3 = (self.C - x + x * y ** 3) ** 2

        loss = self.alpha * loss1 + self.beta * loss2 + self.gamma * loss3 + self.omega

        return self.return_metric(loss)


class HimmelblausFunction(ObjectiveFunction):
    __name__ = "himmelblaus_function"

    def __init__(
        self,
        alpha=1,
        beta=1,
        omega=0,
        A=-11,
        B=-7,
        metric="score",
        input_type="dictionary",
    ):
        super().__init__(metric, input_type)

        self.alpha = alpha
        self.beta = beta
        self.omega = omega

        self.A = A
        self.B = B

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss1 = (x ** 2 + y + self.A) ** 2
        loss2 = (x + y ** 2 + self.B) ** 2

        loss = self.alpha * loss1 + self.beta * loss2 + self.omega

        return self.return_metric(loss)


class HölderTableFunction(ObjectiveFunction):
    __name__ = "hölder_table_function"

    def __init__(
        self,
        alpha=1,
        omega=0,
        A=10,
        angle=1,
        metric="score",
        input_type="dictionary",
    ):
        super().__init__(metric, input_type)

        self.alpha = alpha
        self.omega = omega

        self.A = A
        self.angle = angle

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss1 = np.sin(self.angle * x) * np.cos(self.angle * y)
        loss2 = np.exp(abs(1 - (np.sqrt(x ** 2 + y ** 2) / np.pi)))

        loss = -np.abs(loss1 * loss2) * self.alpha + self.omega

        return self.return_metric(loss)


class CrossInTrayFunction(ObjectiveFunction):
    __name__ = "cross_in_tray_function"

    def __init__(
        self,
        alpha=1,
        omega=0,
        A=-0.0001,
        B=100,
        angle=1,
        metric="score",
        input_type="dictionary",
    ):
        super().__init__(metric, input_type)

        self.alpha = alpha
        self.omega = omega

        self.A = A
        self.B = B
        self.angle = angle

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss1 = np.sin(self.angle * x) * np.sin(self.angle * y)
        loss2 = np.exp(abs(self.B - (np.sqrt(x ** 2 + y ** 2) / np.pi)) + 1)

        loss = (-self.A * (np.abs(loss1 * loss2)) ** 0.1) * self.alpha + self.omega

        return self.return_metric(loss)


class SimionescuFunction(ObjectiveFunction):
    __name__ = "simionescu_function"

    def __init__(
        self,
        alpha=0.1,
        omega=0,
        r_T=1,
        r_S=0.2,
        n=8,
        metric="score",
        input_type="dictionary",
    ):
        super().__init__(metric, input_type)

        self.alpha = alpha
        self.omega = omega

        self.alpha = alpha
        self.r_T = r_T
        self.r_S = r_S
        self.n = n

    def objective_function_dict(self, params):
        x = params["x0"].reshape(-1)
        y = params["x1"].reshape(-1)

        condition = (self.r_T + self.r_S * np.cos(self.n * np.arctan(x / y))) ** 2

        mask = x ** 2 + y ** 2 <= condition
        mask_int = mask.astype(int)

        loss = self.alpha * x * y + self.omega

        loss = mask_int * loss
        loss[~mask] = np.nan

        return self.return_metric(loss)


class EasomFunction(ObjectiveFunction):
    __name__ = "easom_function"

    def __init__(
        self,
        alpha=1,
        omega=0,
        A=-1,
        B=1,
        angle=1,
        metric="score",
        input_type="dictionary",
    ):
        super().__init__(metric, input_type)

        self.alpha = alpha
        self.omega = omega

        self.A = A
        self.B = B
        self.angle = angle

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss1 = self.A * np.cos(x * self.angle) * np.cos(y * self.angle)
        loss2 = np.exp(-((x - np.pi / self.B) ** 2 + (y - np.pi / self.B) ** 2))

        loss = self.alpha * loss1 * loss2 + self.omega

        return self.return_metric(loss)


class EggholderFunction(ObjectiveFunction):
    __name__ = "eggholder_function"

    def __init__(
        self,
        alpha=1,
        beta=1,
        omega=0,
        A=-1,
        B=47,
        angle=1,
        metric="score",
        input_type="dictionary",
    ):
        super().__init__(metric, input_type)

        self.alpha = alpha
        self.beta = beta
        self.omega = omega

        self.A = A
        self.B = B
        self.angle = angle

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss1 = self.A * (y + self.B)
        loss2 = np.sin(self.angle * np.sqrt(np.abs(x / 2 + (y + self.B))))
        loss3 = self.A * x * np.sin(self.angle * np.sqrt(np.abs(x - (y + self.B))))

        loss = self.alpha * loss1 * loss2 + self.beta * loss3 + self.omega

        return self.return_metric(loss)
