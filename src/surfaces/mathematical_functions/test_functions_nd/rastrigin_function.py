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

    def create_objective_function(self):
        def rastrigin_function(params):
            loss = 0
            for dim in range(self.n_dim):
                dim_str = "x" + str(dim)
                x = params[dim_str]

                loss += x * x - self.A * np.cos(self.angle * x)

            return self.A * self.n_dim + loss

        self.pure_objective_function = rastrigin_function

    def search_space(self, min=-5, max=5, step=0.1, value_types="array"):
        return super().create_n_dim_search_space(min, max, step, value_types)
