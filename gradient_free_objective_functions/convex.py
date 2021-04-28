# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from base_objective_function import ObjectiveFunction


class SphereFunction(ObjectiveFunction):
    def __init__(self, n_dim, A=1, metric="score"):
        super().__init__()
        self.__name__ = "sphere_function"

        self.n_dim = n_dim
        self.A = A

    def _objective_function(self, params):
        loss = 0
        for dim in range(self.n_dim):
            dim_str = "x" + str(dim)
            x = params[dim_str]

            loss += self.A * x * x

        if self.metric == "score":
            return -loss
        elif self.metric == "loss":
            return loss

    def __call__(self, params):
        return self._objective_function(params)
