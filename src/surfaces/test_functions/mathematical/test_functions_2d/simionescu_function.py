# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_mathematical_function import MathematicalFunction


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
        sleep=0,
    ):
        super().__init__(metric, sleep)
        self.n_dim = 2

        self.A = A
        self.r_T = r_T
        self.r_S = r_S
        self.n = n

    def create_objective_function(self):
        def simionescu_function(params):
            x = params["x0"].reshape(-1)
            y = params["x1"].reshape(-1)

            condition = (self.r_T + self.r_S * np.cos(self.n * np.arctan(x / y))) ** 2

            mask = x**2 + y**2 <= condition
            mask_int = mask.astype(int)

            loss = self.A * x * y
            loss = mask_int * loss
            loss[~mask] = np.nan

            return loss

        self.pure_objective_function = simionescu_function

    def search_space(self, min=-1.25, max=1.25, value_types="array", size=10000):
        return super().create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
