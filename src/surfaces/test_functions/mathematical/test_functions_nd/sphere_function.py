# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .._base_mathematical_function import MathematicalFunction


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

    def __init__(self, n_dim, A=1, metric="score", sleep=0):
        super().__init__(metric, sleep)
        self.n_dim = n_dim
        self.A = A

    def create_objective_function(self):
        def sphere_function(params):
            loss = 0
            for dim in range(self.n_dim):
                dim_str = "x" + str(dim)
                x = params[dim_str]

                loss += self.A * x * x

            return loss

        self.pure_objective_function = sphere_function

    def search_space(self, min=-5, max=5, size=10000, value_types="array"):
        return super().create_n_dim_search_space(
            min, max, size=size, value_types=value_types
        )
