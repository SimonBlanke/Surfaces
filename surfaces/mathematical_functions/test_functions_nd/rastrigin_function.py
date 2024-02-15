# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_objective_function import MathematicalFunction


class RastriginFunction(MathematicalFunction):
    name = "Rastrigin Function"
    _name_ = "rastrigin_function"
    __name__ = "RastriginFunction"

    explanation = """
    
    """

    reference = """
    
    """

    dimensions = "n"
    formula = r"""f(\vec{x}) = An + \sum^n_{i=1} \left[x_i^2 - A\cos(2\pi x_i)\right]
                \newline
                \text{where:} A = 10"""
    global_minimum = r"""f(\vec x = 0) = 0"""

    def __init__(
        self,
        n_dim,
        A=10,
        angle=2 * np.pi,
        metric="score",
        input_type="dictionary",
        sleep=0,
    ):
        super().__init__(metric, input_type, sleep)

        self.n_dim = n_dim
        self.A = A
        self.angle = angle

    def objective_function_dict(self, params):
        loss = 0
        for dim in range(self.n_dim):
            dim_str = "x" + str(dim)
            x = params[dim_str]

            loss += x * x - self.A * np.cos(self.angle * x)

        loss = self.A * self.n_dim + loss

        return self.return_metric(loss)

    def search_space(self, value_types="array", steps=100):
        min_x0 = -6
        min_x1 = -6

        max_x0 = 6
        max_x1 = 6

        step_size_x0 = (max_x0 - min_x0) / steps
        step_size_x1 = (max_x1 - min_x1) / steps

        return super().search_space(
            search_space_blank={
                "x0": (min_x0, max_x0, step_size_x0),
                "x1": (min_x1, max_x1, step_size_x1),
            },
            value_types=value_types,
        )
