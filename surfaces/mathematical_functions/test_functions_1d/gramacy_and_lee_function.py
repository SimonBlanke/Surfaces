# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_objective_function import MathematicalFunction


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

    def __init__(self, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 1

    def objective_function_dict(self, params):
        x = params["x0"]

        loss = (np.sin(10 * np.pi * x) / (2 * x)) + (x - 1) ** 4

        return self.return_metric(loss)

    def search_space(self, value_types="array", steps=100):
        min_x0 = 0.5
        max_x0 = 2.5
        step_size_x0 = (max_x0 - min_x0) / steps

        return super().search_space(
            search_space_blank={
                "x0": (min_x0, max_x0, step_size_x0),
            },
            value_types=value_types,
        )
