# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from .._base_mathematical_function import MathematicalFunction


class AckleyFunction(MathematicalFunction):
    name = "Ackley Function"
    _name_ = "ackley_function"
    __name__ = "AckleyFunction"

    explanation = """
    The Ackley function is a non-convex function used as a performance test problem for optimization algorithms.
    """

    reference = """
    Ackley, D. H. (1987) "A connectionist machine for genetic hillclimbing", Kluwer Academic Publishers, Boston MA.
    """

    dimensions = "2"
    formula = r"f(x, y) = -20 \exp\left[-0.2\sqrt{0.5(x^2+y^2)} \right] -\exp\left[ 0.5(\cos2\pi x + \cos2\pi y) \right] + e + 20"
    global_minimum = r"f(\vec{x}=0) = 0"

    def __init__(
        self,
        A=20,
        angle=2 * np.pi,
        metric="score",
        sleep=0,
    ):
        super().__init__(metric, sleep)

        self.n_dim = 2

        self.A = A
        self.angle = angle

    def create_objective_function(self):
        def ackley_function(params):
            x = params["x0"]
            y = params["x1"]

            loss1 = -self.A * np.exp(-0.2 * np.sqrt(0.5 * (x * x + y * y)))
            loss2 = -np.exp(0.5 * (np.cos(self.angle * x) + np.cos(self.angle * y)))
            loss3 = np.exp(1)
            loss4 = self.A

            return loss1 + loss2 + loss3 + loss4

        self.pure_objective_function = ackley_function

    def search_space(self, min=-5, max=5, value_types="array", size=10000):
        return super().create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
