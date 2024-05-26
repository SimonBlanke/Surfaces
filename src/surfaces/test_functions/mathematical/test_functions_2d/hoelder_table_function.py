# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_mathematical_function import MathematicalFunction


class HölderTableFunction(MathematicalFunction):
    name = "Hölder Table Function"
    _name_ = "hölder_table_function"
    __name__ = "HölderTableFunction"

    explanation = """
    
    """

    reference = """
    
    """

    dimensions = "2"
    formula = r"""f(x,y) = - \left|\sin x \cos y \exp \left(\left|1 - \frac{\sqrt{x^{2} + y^{2}}}{\pi} \right|\right)\right|"""
    global_minimum = r"""      
      f(8.05502,  9.66459) = -19.2085 \\
      f(-8.05502,  9.66459) = -19.2085 \\
      f(8.05502,-9.66459) = -19.2085 \\
      f(-8.05502,-9.66459) = -19.2085
      """

    def __init__(self, A=10, angle=1, metric="score", sleep=0):
        super().__init__(metric, sleep)
        self.n_dim = 2

        self.A = A
        self.angle = angle

    def create_objective_function(self):
        def hölder_table_function(params):
            x = params["x0"]
            y = params["x1"]

            loss1 = np.sin(self.angle * x) * np.cos(self.angle * y)
            loss2 = np.exp(abs(1 - (np.sqrt(x**2 + y**2) / np.pi)))

            return -np.abs(loss1 * loss2)

        self.pure_objective_function = hölder_table_function

    def search_space(self, min=-10, max=10, value_types="array", size=10000):
        return super().create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
