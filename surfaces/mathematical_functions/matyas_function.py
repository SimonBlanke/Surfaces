# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ._base_objective_function import ObjectiveFunction


class MatyasFunction(ObjectiveFunction):
    name = "Matyas Function"
    _name_ = "matyas_function"
    __name__ = "MatyasFunction"

    def __init__(self, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y

        return self.return_metric(loss)
