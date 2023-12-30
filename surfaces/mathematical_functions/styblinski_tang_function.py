# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ._base_objective_function import MathematicalFunction


class StyblinskiTangFunction(MathematicalFunction):
    name = "Styblinski Tang Function"
    _name_ = "styblinski_tang_function"
    __name__ = "StyblinskiTangFunction"

    formula = (
        r"""f(\vec {x}) = \frac{\sum_{i=1}^{n} x_{i}^{4} - 16x_{i}^{2} + 5x_{i}}{2}"""
    )

    def __init__(self, n_dim, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = n_dim

    def objective_function_dict(self, params):
        loss = 0
        for dim in range(self.n_dim):
            dim_str = "x" + str(dim)
            x = params[dim_str]

            loss += x**4 - 16 * x**2 + 5 * x

        loss = loss / 2

        return self.return_metric(loss)
