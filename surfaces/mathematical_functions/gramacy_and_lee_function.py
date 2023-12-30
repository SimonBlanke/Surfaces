# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from ._base_objective_function import ObjectiveFunction


class GramacyAndLeeFunction(ObjectiveFunction):
    name = "Gramacy And Lee Function"
    _name_ = "gramacy_and_lee_function"
    __name__ = "GramacyAndLeeFunction"

    def __init__(self, *args, **kwargs):
        super().__init__(args, **kwargs)
        self.n_dim = 1

    def objective_function_dict(self, params):
        x = params["x0"]

        loss = np.sin(10 * np.pi * x) / (2 * x) + (x - 1) ** 4

        return self.return_metric(loss)
