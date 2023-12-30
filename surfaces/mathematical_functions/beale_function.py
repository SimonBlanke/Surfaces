# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ._base_objective_function import MathematicalFunction


class BealeFunction(MathematicalFunction):
    name = "Beale Function"
    _name_ = "beale_function"
    __name__ = "BealeFunction"

    dimensions = "2"
    formula = (
        r"""f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2"""
    )
    global_minimum = r"""f(3, 0.5) = 0"""

    def __init__(
        self, A=1.5, B=2.25, C=2.652, metric="score", input_type="dictionary", sleep=0
    ):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

        self.A = A
        self.B = B
        self.C = C

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss1 = (self.A - x + x * y) ** 2
        loss2 = (self.B - x + x * y**2) ** 2
        loss3 = (self.C - x + x * y**3) ** 2

        loss = loss1 + loss2 + loss3

        return self.return_metric(loss)
