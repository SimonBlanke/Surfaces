# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from sympy import Sum, Indexed
from sympy.abc import x, i


from .base_objective_function import ObjectiveFunction


class SphereFunction(ObjectiveFunction):
    name = "sphere_function"
    continuous = True
    convex = True
    separable = True
    differentiable = False
    mutimodal = False
    randomized_term = False
    parametric = False

    def __init__(self, n_dim, A=1, metric="score", input_type="dictionary"):
        super().__init__(metric, input_type)
        self.equation = Sum(Indexed(x, i) ** 2, (i, 1, n_dim))

        self.n_dim = n_dim
        self.A = A

    def objective_function_dict(self, params):
        loss = 0
        for dim in range(self.n_dim):
            dim_str = "x" + str(dim)
            x = params[dim_str]

            loss += self.A * x * x

        return self.return_metric(loss)
