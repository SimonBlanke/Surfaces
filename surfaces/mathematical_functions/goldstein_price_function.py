# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ._base_objective_function import ObjectiveFunction


class GoldsteinPriceFunction(ObjectiveFunction):
    name = "Goldstein Price Function"
    _name_ = "goldstein_price_function"
    __name__ = "GoldsteinPriceFunction"

    def __init__(self, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss1 = 1 + (x + y + 1) ** 2 * (
            19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2
        )
        loss2 = 30 + (2 * x - 3 * y) ** 2 * (
            18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2
        )

        loss = loss1 * loss2

        return self.return_metric(loss)
