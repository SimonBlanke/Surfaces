# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from ._base_objective_function import MathematicalFunction


class ThreeHumpCamelFunction(MathematicalFunction):
    name = "Three Hump Camel Function"
    _name_ = "three_hump_camel_function"
    __name__ = "ThreeHumpCamelFunction"

    dimensions = "2"
    formula = r"""f(x,y) = -\cos \left(x\right)\cos \left(y\right) \exp\left[-\left(\left(x-\pi\right)^{2} + \left(y-\pi\right)^{2}\right)\right]"""
    global_minimum = r"""f(0,0)= 0"""

    def __init__(self, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss = 2 * x**2 - 1.05 * x**4 + x**6 / 6 + x * y + y**2

        return self.return_metric(loss)
