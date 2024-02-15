# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .._base_objective_function import MathematicalFunction


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

    def __init__(self, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss = 0.26 * (x**2 + y**2) - 0.48 * x * y

        return self.return_metric(loss)

    def search_space(self, value_types="array", steps=100):
        min_x0 = -10
        min_x1 = -10

        max_x0 = 10
        max_x1 = 10

        step_size_x0 = (max_x0 - min_x0) / steps
        step_size_x1 = (max_x1 - min_x1) / steps

        return super().search_space(
            search_space_blank={
                "x0": (min_x0, max_x0, step_size_x0),
                "x1": (min_x1, max_x1, step_size_x1),
            },
            value_types=value_types,
        )
