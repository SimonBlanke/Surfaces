# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_objective_function import MathematicalFunction


class EasomFunction(MathematicalFunction):
    name = "Easom Function"
    _name_ = "easom_function"
    __name__ = "EasomFunction"

    explanation = """
    
    """

    reference = """
    
    """

    dimensions = "2"
    formula = r"""f(x, y) = -\cos (x) \cos (y) \ exp\left[- \left((x-\pi)^2 + (y-\pi)^2 \right) \right]"""
    global_minimum = r"""f(\pi, \pi) = -1"""

    def __init__(
        self, A=-1, B=1, angle=1, metric="score", input_type="dictionary", sleep=0
    ):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

        self.A = A
        self.B = B
        self.angle = angle

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss1 = self.A * np.cos(x * self.angle) * np.cos(y * self.angle)
        loss2 = np.exp(-((x - np.pi / self.B) ** 2 + (y - np.pi / self.B) ** 2))

        loss = loss1 * loss2

        return self.return_metric(loss)

    def search_space(self, value_types="array", steps=100):
        min_x0 = -10
        min_x1 = -10

        max_x0 = 10
        max_x1 = 10

        step_size_x0 = (max_x0 - min_x0) / steps
        step_size_x1 = (max_x1 - min_x1) / steps

        return super().search_space(
            search_space_blank={
                "x0": (min_x0, max_x0, step_size_x0),
                "x1": (min_x1, max_x1, step_size_x1),
            },
            value_types=value_types,
        )
