# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ._base_objective_function import MathematicalFunction


class BoothFunction(MathematicalFunction):
    name = "Booth Function"
    _name_ = "booth_function"
    __name__ = "BoothFunction"

    dimensions = "2"
    formula = r"""f(x,y) = (x + 2y - 7)^2 + (2x + y - 5)^2"""
    global_minimum = r"""f(1,3)=0"""

    def __init__(self, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss1 = (x + 2 * y - 7) ** 2
        loss2 = (2 * x + y - 5) ** 2

        loss = loss1 * loss2

        return self.return_metric(loss)
