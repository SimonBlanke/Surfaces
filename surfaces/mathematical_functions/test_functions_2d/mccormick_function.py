# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_objective_function import MathematicalFunction


class McCormickFunction(MathematicalFunction):
    name = "Mc Cormick Function"
    _name_ = "mccormick_function"
    __name__ = "McCormickFunction"

    explanation = """
    
    """

    reference = """
    
    """

    dimensions = "2"
    formula = (
        r"""f(x,y) = \sin \left(x+y\right) + \left(x-y\right)^{2} - 1.5x + 2.5y + 1"""
    )
    global_minimum = r"""f(-0.54719, -1.54719) = -1.9133"""

    def __init__(self, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss = np.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1

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
