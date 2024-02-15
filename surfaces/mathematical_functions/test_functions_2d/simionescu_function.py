# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_objective_function import MathematicalFunction


class SimionescuFunction(MathematicalFunction):
    name = "Simionescu Function"
    _name_ = "simionescu_function"
    __name__ = "SimionescuFunction"

    explanation = """
    
    """

    reference = """
    
    """

    formula = r"""f(x,y) = 0.1xy 
            \newline
            x^2+y^2\le\left[r_{T}+r_{S}\cos\left(n \arctan \frac{x}{y} \right)\right]^2
            \newline
            \text{where: }  r_{T}=1, r_{S}=0.2 \text{ and } n = 8"""
    global_minimum = r"""f(\pm 0.84852813,\mp 0.84852813) = -0.072"""

    def __init__(
        self,
        A=0.1,
        r_T=1,
        r_S=0.2,
        n=8,
        metric="score",
        input_type="dictionary",
        sleep=0,
    ):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

        self.A = A
        self.r_T = r_T
        self.r_S = r_S
        self.n = n

    def objective_function_dict(self, params):
        x = params["x0"].reshape(-1)
        y = params["x1"].reshape(-1)

        condition = (self.r_T + self.r_S * np.cos(self.n * np.arctan(x / y))) ** 2

        mask = x**2 + y**2 <= condition
        mask_int = mask.astype(int)

        loss = self.A * x * y
        loss = mask_int * loss
        loss[~mask] = np.nan

        return self.return_metric(loss)

    def search_space(self, value_types="array", steps=100):
        min_x0 = -1.25
        min_x1 = -1.25

        max_x0 = 1.25
        max_x1 = 1.25

        step_size_x0 = (max_x0 - min_x0) / steps
        step_size_x1 = (max_x1 - min_x1) / steps

        return super().search_space(
            search_space_blank={
                "x0": (min_x0, max_x0, step_size_x0),
                "x1": (min_x1, max_x1, step_size_x1),
            },
            value_types=value_types,
        )
