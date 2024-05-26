# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .._base_mathematical_function import MathematicalFunction


class MatyasFunction(MathematicalFunction):
    name = "Matyas Function"
    _name_ = "matyas_function"
    __name__ = "MatyasFunction"

    explanation = """
    
    """

    reference = """
    
    """

    dimensions = "2"
    formula = r"""f(x,y) = 0.26 \left( x^{2} + y^{2}\right) - 0.48 xy"""
    global_minimum = r"""f(0,0) = 0"""

    def __init__(self, metric="score", sleep=0):
        super().__init__(metric, sleep)
        self.n_dim = 2

    def create_objective_function(self):
        def matyas_function(params):
            x = params["x0"]
            y = params["x1"]

            return 0.26 * (x**2 + y**2) - 0.48 * x * y

        self.pure_objective_function = matyas_function

    def search_space(self, min=-10, max=10, value_types="array", size=10000):
        return super().create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
