# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ._base_objective_function import MathematicalFunction


class HimmelblausFunction(MathematicalFunction):
    name = "Himmelblau's Function"
    _name_ = "himmelblaus_function"
    __name__ = "HimmelblausFunction"

    formula = r"""f(x, y) = (x^2+y-11)^2 + (x+y^2-7)^2"""

    def __init__(self, A=-11, B=-7, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

        self.A = A
        self.B = B

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss1 = (x**2 + y + self.A) ** 2
        loss2 = (x + y**2 + self.B) ** 2

        loss = loss1 + loss2

        return self.return_metric(loss)
