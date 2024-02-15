# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_objective_function import MathematicalFunction


class SchafferFunctionN2(MathematicalFunction):
    name = "Schaffer Function N2"
    _name_ = "schaffer_function_n2"
    __name__ = "SchafferFunctionN2"

    explanation = """
    
    """

    reference = """
    
    """

    dimensions = "2"
    formula = r"""f(x,y) = 0.5 + \frac{\sin^{2}\left(x^{2} - y^{2}\right) - 0.5}{\left[1 + 0.001\left(x^{2} + y^{2}\right) \right]^{2}}"""
    global_minimum = r"""f(0,0) = 0"""

    def __init__(self, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss = 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)

        return self.return_metric(loss)

    def search_space(self, value_types="array", steps=100):
        min_x0 = -50
        min_x1 = -50

        max_x0 = 50
        max_x1 = 50

        step_size_x0 = (max_x0 - min_x0) / steps
        step_size_x1 = (max_x1 - min_x1) / steps

        return super().search_space(
            search_space_blank={
                "x0": (min_x0, max_x0, step_size_x0),
                "x1": (min_x1, max_x1, step_size_x1),
            },
            value_types=value_types,
        )
