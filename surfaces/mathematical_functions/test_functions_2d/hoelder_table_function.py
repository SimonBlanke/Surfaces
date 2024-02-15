# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_objective_function import MathematicalFunction


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

    def __init__(self, A=10, angle=1, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

        self.A = A
        self.angle = angle

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss1 = np.sin(self.angle * x) * np.cos(self.angle * y)
        loss2 = np.exp(abs(1 - (np.sqrt(x**2 + y**2) / np.pi)))

        loss = -np.abs(loss1 * loss2)

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
