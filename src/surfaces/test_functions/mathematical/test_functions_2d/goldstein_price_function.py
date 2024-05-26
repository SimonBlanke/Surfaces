# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .._base_mathematical_function import MathematicalFunction


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

    def __init__(self, metric="score", sleep=0):
        super().__init__(metric, sleep)
        self.n_dim = 2

    def create_objective_function(self):
        def goldstein_price_function(params):
            x = params["x0"]
            y = params["x1"]

            loss1 = 1 + (x + y + 1) ** 2 * (
                19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2
            )
            loss2 = 30 + (2 * x - 3 * y) ** 2 * (
                18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2
            )

            return loss1 * loss2

        self.pure_objective_function = goldstein_price_function

    def search_space(self, min=-2, max=2, value_types="array", size=10000):
        return super().create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
