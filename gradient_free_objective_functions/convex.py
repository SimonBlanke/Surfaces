# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .base_objective_function import ObjectiveFunction


class SphereFunction(ObjectiveFunction):
    def __init__(self, n_dim, A=1, metric="score", parameter_type="dictionary"):
        super().__init__(metric, parameter_type)
        self.__name__ = "sphere_function"

        self.n_dim = n_dim
        self.A = A

    def objective_function_dict(self, params):
        loss = 0
        for dim in range(self.n_dim):
            dim_str = "x" + str(dim)
            x = params[dim_str]

            loss += self.A * x * x

        return self.return_metric(loss)
