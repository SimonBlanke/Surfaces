# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .._base_mathematical_function import MathematicalFunction


class BealeFunction(MathematicalFunction):
    name = "Beale Function"
    _name_ = "beale_function"
    __name__ = "BealeFunction"

    explanation = """
    The Beale function is multimodal, with sharp peaks at the corners of the input domain.
    """

    reference = """
    Global Optimization Test Problems. Retrieved June 2013, from
http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO
    """

    dimensions = "2"
    formula = (
        r"""f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2"""
    )
    global_minimum = r"""f(3, 0.5) = 0"""

    def __init__(self, A=1.5, B=2.25, C=2.652, metric="score", sleep=0):
        super().__init__(metric, sleep)
        self.n_dim = 2

        self.A = A
        self.B = B
        self.C = C

    def create_objective_function(self):
        def beale_function(params):
            x = params["x0"]
            y = params["x1"]

            loss1 = (self.A - x + x * y) ** 2
            loss2 = (self.B - x + x * y**2) ** 2
            loss3 = (self.C - x + x * y**3) ** 2

            return loss1 + loss2 + loss3

        self.pure_objective_function = beale_function

    def search_space(self, min=-4.5, max=4.5, value_types="array", size=10000):
        return super().create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
