# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_objective_function import MathematicalFunction


class EggholderFunction(MathematicalFunction):
    name = "Eggholder Function"
    _name_ = "eggholder_function"
    __name__ = "EggholderFunction"

    explanation = """
    
    """

    reference = """
    
    """

    dimensions = "2"
    formula = r"""f(x, y) = - (y + 47) \sin \sqrt{\left| \frac{x}{2} + (y + 47) \right|} - x \sin \sqrt{\left| x- (y + 47) \right|}"""
    global_minimum = r"""f(512, 404.2319) = -959.6407"""

    def __init__(self, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

    def create_objective_function(self):
        def eggholder_function(params):
            x = params["x0"]
            y = params["x1"]

            return (y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47)))) - x * np.sin(
                np.sqrt(np.abs(x - (y + 47)))
            )

        self.pure_objective_function = eggholder_function

    def search_space(self, value_types="array", steps=100):
        min_x0 = -1000
        min_x1 = -1000

        max_x0 = 1000
        max_x1 = 1000

        step_size_x0 = (max_x0 - min_x0) / steps
        step_size_x1 = (max_x1 - min_x1) / steps

        return super().search_space_from_blank(
            search_space_blank={
                "x0": (min_x0, max_x0, step_size_x0),
                "x1": (min_x1, max_x1, step_size_x1),
            },
            value_types=value_types,
        )
