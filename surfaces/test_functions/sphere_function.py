# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .base_objective_function import ObjectiveFunction


class SphereFunction(ObjectiveFunction):
    name = "Sphere Function"
    _name_ = "sphere_function"
    __name__ = "SphereFunction"

    def __init__(self, n_dim, A=1, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = n_dim
        self.A = A

    def objective_function_dict(self, params):
        loss = 0
        for dim in range(self.n_dim):
            dim_str = "x" + str(dim)
            x = params[dim_str]

            loss += self.A * x * x

        return self.return_metric(loss)
