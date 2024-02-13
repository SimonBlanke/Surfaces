# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .._base_objective_function import MathematicalFunction


class SphereFunction(MathematicalFunction):
    name = "Sphere Function"
    _name_ = "sphere_function"
    __name__ = "SphereFunction"

    explanation = """The Sphere function has d local minima except for the global one. It is continuous, convex and unimodal. The plot shows its two-dimensional form."""

    reference = """
    
    """

    dimensions = "n"
    formula = r"f(\vec{x}) = \sum^n_{i=1}x^2_i"
    global_minimum = r"f(\vec{x}=0) = 0"

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
