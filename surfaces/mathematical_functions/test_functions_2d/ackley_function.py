# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_objective_function import MathematicalFunction


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
        self, A=20, angle=2 * np.pi, metric="score", input_type="dictionary", sleep=0
    ):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

        self.A = A
        self.angle = angle

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss1 = -self.A * np.exp(-0.2 * np.sqrt(0.5 * (x * x + y * y)))
        loss2 = -np.exp(0.5 * (np.cos(self.angle * x) + np.cos(self.angle * y)))
        loss3 = np.exp(1)
        loss4 = self.A

        loss = loss1 + loss2 + loss3 + loss4

        return self.return_metric(loss)

    def search_space(self, value_types="array", steps=100):
        min_x0 = -5
        min_x1 = -5

        max_x0 = 5
        max_x1 = 5

        step_size_x0 = (max_x0 - min_x0) / steps
        step_size_x1 = (max_x1 - min_x1) / steps

        return super().search_space(
            search_space_blank={
                "x0": (min_x0, max_x0, step_size_x0),
                "x1": (min_x1, max_x1, step_size_x1),
            },
            value_types=value_types,
        )
