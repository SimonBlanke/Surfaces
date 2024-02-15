# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .._base_objective_function import MathematicalFunction


class GoldsteinPriceFunction(MathematicalFunction):
    name = "Goldstein Price Function"
    _name_ = "goldstein_price_function"
    __name__ = "GoldsteinPriceFunction"

    explanation = """
    
    """

    reference = """
    
    """

    dimensions = "2"
    formula = r"""f(x,y) = \left[1+\left(x+y+1\right)^{2}\left(19-14x+3x^{2}-14y+6xy+3y^{2}\right)\right]\left[30+\left(2x-3y\right)^{2}\left(18-32x+12x^{2}+48y-36xy+27y^{2}\right)\right]"""
    global_minimum = r"""f(0,-1) = 3"""

    def __init__(self, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss1 = 1 + (x + y + 1) ** 2 * (
            19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2
        )
        loss2 = 30 + (2 * x - 3 * y) ** 2 * (
            18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2
        )

        loss = loss1 * loss2

        return self.return_metric(loss)

    def search_space(self, value_types="array", steps=100):
        min_x0 = -2
        min_x1 = -2

        max_x0 = 2
        max_x1 = 2

        step_size_x0 = (max_x0 - min_x0) / steps
        step_size_x1 = (max_x1 - min_x1) / steps

        return super().search_space(
            search_space_blank={
                "x0": (min_x0, max_x0, step_size_x0),
                "x1": (min_x1, max_x1, step_size_x1),
            },
            value_types=value_types,
        )
