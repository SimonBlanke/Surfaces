# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from ._base_objective_function import ObjectiveFunction


class EggholderFunction(ObjectiveFunction):
    name = "Eggholder Function"
    _name_ = "eggholder_function"
    __name__ = "EggholderFunction"

    def __init__(self, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss = (y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47)))) - x * np.sin(
            np.sqrt(np.abs(x - (y + 47)))
        )

        return self.return_metric(loss)
