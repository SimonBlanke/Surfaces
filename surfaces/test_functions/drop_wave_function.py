# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from ._base_objective_function import ObjectiveFunction


class DropWaveFunction(ObjectiveFunction):
    name = "Drop Wave Function"
    _name_ = "drop_wave_function"
    __name__ = "DropWaveFunction"

    def __init__(self, *args, **kwargs):
        super().__init__(args, **kwargs)
        self.n_dim = 2

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss = -(1 + np.cos(12 * np.sqrt(x ** 2 + y ** 2))) / (
            0.5 * (x ** 2 + y ** 2) + 2
        )

        return self.return_metric(loss)
