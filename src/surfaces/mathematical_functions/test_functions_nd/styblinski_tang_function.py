# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .._base_objective_function import MathematicalFunction


class StyblinskiTangFunction(MathematicalFunction):
    name = "Styblinski Tang Function"
    _name_ = "styblinski_tang_function"
    __name__ = "StyblinskiTangFunction"

    explanation = """
    
    """

    reference = """
    
    """

    dimensions = "n"
    formula = (
        r"""f(\vec {x}) = \frac{\sum_{i=1}^{n} x_{i}^{4} - 16x_{i}^{2} + 5x_{i}}{2}"""
    )
    global_minimum = r"""-39.16617n < f(\underbrace{-2.903534, \ldots, -2.903534}_{n \text{ times}} ) < -39.16616n"""

    def __init__(self, n_dim, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = n_dim

    def create_objective_function(self):
        def styblinski_tang_function(params):
            loss = 0
            for dim in range(self.n_dim):
                dim_str = "x" + str(dim)
                x = params[dim_str]

                loss += x**4 - 16 * x**2 + 5 * x

            return loss / 2

        self.pure_objective_function = styblinski_tang_function

    def search_space(self, min=-5, max=5, step=0.1, value_types="array"):
        return super().create_n_dim_search_space(min, max, step, value_types)
