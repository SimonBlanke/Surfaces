# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_mathematical_function import MathematicalFunction


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

    def __init__(self, metric="score", sleep=0):
        super().__init__(metric, sleep)
        self.n_dim = 2

    def create_objective_function(self):
        def eggholder_function(params):
            x = params["x0"]
            y = params["x1"]

            return (y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47)))) - x * np.sin(
                np.sqrt(np.abs(x - (y + 47)))
            )

        self.pure_objective_function = eggholder_function

    def search_space(self, min=-1000, max=1000, value_types="array", size=10000):
        return super().create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
