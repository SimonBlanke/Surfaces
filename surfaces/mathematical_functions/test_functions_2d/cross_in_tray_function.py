# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_objective_function import MathematicalFunction


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
        input_type="dictionary",
        sleep=0,
    ):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

        self.A = A
        self.B = B
        self.angle = angle

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss1 = np.sin(self.angle * x) * np.sin(self.angle * y)
        loss2 = np.exp(abs(self.B - (np.sqrt(x**2 + y**2) / np.pi)) + 1)

        loss = -self.A * (np.abs(loss1 * loss2)) ** 0.1

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
