# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .._base_objective_function import MathematicalFunction


class RosenbrockFunction(MathematicalFunction):
    name = "Rosenbrock Function"
    _name_ = "rosenbrock_function"
    __name__ = "RosenbrockFunction"

    explanation = """
    
    """

    reference = """
    
    """

    dimensions = "2"
    formula = r"""f(\vec {x}) = \sum_{i=1}^{n-1} \left[ 100 \left(x_{i+1} - x_{i}^{2}\right)^{2} + \left(1 - x_{i}\right)^{2}\right]"""
    global_minimum = r"""f(1,1) = 0"""

    def __init__(self, A=1, B=100, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

        self.A = A
        self.B = B

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss = (self.A - x) ** 2 + self.B * (y - x**2) ** 2

        return self.return_metric(loss)
