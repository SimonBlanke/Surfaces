# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .._base_mathematical_function import MathematicalFunction


class HimmelblausFunction(MathematicalFunction):
    name = "Himmelblau's Function"
    _name_ = "himmelblaus_function"
    __name__ = "HimmelblausFunction"

    explanation = """
    
    """

    reference = """
    
    """

    dimensions = "2"
    formula = r"""f(x, y) = (x^2+y-11)^2 + (x+y^2-7)^2"""
    global_minimum = r"""
      f(3.0,  2.0) = 0.0 \\
      f(-2.805118, 3.131312) = 0.0 \\
      f(-3.779310, -3.283186) = 0.0 \\
      f(3.584428, -1.848126) = 0.0
    """

    def __init__(self, A=-11, B=-7, metric="score", sleep=0):
        super().__init__(metric, sleep)
        self.n_dim = 2

        self.A = A
        self.B = B

    def create_objective_function(self):
        def himmelblaus_function(params):
            x = params["x0"]
            y = params["x1"]

            loss1 = (x**2 + y + self.A) ** 2
            loss2 = (x + y**2 + self.B) ** 2

            return loss1 + loss2

        self.pure_objective_function = himmelblaus_function

    def search_space(self, min=-5, max=5, value_types="array", size=10000):
        return super().create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
