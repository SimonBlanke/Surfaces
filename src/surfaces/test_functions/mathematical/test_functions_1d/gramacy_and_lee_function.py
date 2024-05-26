# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_mathematical_function import MathematicalFunction


class GramacyAndLeeFunction(MathematicalFunction):
    name = "Gramacy And Lee Function"
    _name_ = "gramacy_and_lee_function"
    __name__ = "GramacyAndLeeFunction"

    explanation = """
    This is a simple one-dimensional test function.
    """

    reference = """
    Gramacy, R. B., & Lee, H. K. (2012). Cases for the nugget in modeling computer experiments. Statistics and Computing, 22(3), 713-722.
    """

    dimensions = "1"
    formula = r"""f(x) = \frac{\sin(10\pi x)}{2x} + (x - 1)^4"""
    global_minimum = r"""-"""

    def __init__(self, metric="score", sleep=0):
        super().__init__(metric, sleep)
        self.n_dim = 1

    def create_objective_function(self):
        def gramacy_and_lee_function(params):
            x = params["x0"]

            return (np.sin(10 * np.pi * x) / (2 * x)) + (x - 1) ** 4

        self.pure_objective_function = gramacy_and_lee_function

    def search_space(self, min=0.5, max=2.5, value_types="array", size=10000):
        return super().create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
