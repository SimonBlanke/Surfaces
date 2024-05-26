# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_mathematical_function import MathematicalFunction


class CrossInTrayFunction(MathematicalFunction):
    name = "Cross In Tray Function"
    _name_ = "cross_in_tray_function"
    __name__ = "CrossInTrayFunction"

    explanation = """
    
    """

    reference = """
    
    """

    dimensions = "2"
    formula = r"""f(x,y) = -0.0001 \left[ \left| \sin x \sin y \exp\left( \left| 100 - \frac{\sqrt{x^2+y^2}}{\pi} \right| \right)  \right| +1  \right]^{0.1}"""
    global_minimum = r"""
    f(1.34941, -1.34941) = -2.06261 \\
    f(1.34941, 1.34941) = -2.06261 \\
    f(-1.34941, 1.34941) = -2.06261 \\
    f(-1.34941, -1.34941) = -2.06261 \\
        """

    def __init__(
        self,
        A=-0.0001,
        B=100,
        angle=1,
        metric="score",
        sleep=0,
    ):
        super().__init__(metric, sleep)
        self.n_dim = 2

        self.A = A
        self.B = B
        self.angle = angle

    def create_objective_function(self):
        def cross_in_tray_function(params):
            x = params["x0"]
            y = params["x1"]

            loss1 = np.sin(self.angle * x) * np.sin(self.angle * y)
            loss2 = np.exp(abs(self.B - (np.sqrt(x**2 + y**2) / np.pi)) + 1)

            return -self.A * (np.abs(loss1 * loss2)) ** 0.1

        self.pure_objective_function = cross_in_tray_function

    def search_space(self, min=-10, max=10, value_types="array", size=10000):
        return super().create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
