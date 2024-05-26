# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .._base_mathematical_function import MathematicalFunction


class BoothFunction(MathematicalFunction):
    name = "Booth Function"
    _name_ = "booth_function"
    __name__ = "BoothFunction"

    explanation = """
    
    """

    reference = """
    Global Optimization Test Problems. Retrieved June 2013, from
http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO
    """

    dimensions = "2"
    formula = r"""f(x,y) = (x + 2y - 7)^2 + (2x + y - 5)^2"""
    global_minimum = r"""f(1,3)=0"""

    def __init__(self, metric="score", sleep=0):
        super().__init__(metric, sleep)
        self.n_dim = 2

    def create_objective_function(self):

        def booth_function(params):
            x = params["x0"]
            y = params["x1"]

            loss1 = (x + 2 * y - 7) ** 2
            loss2 = (2 * x + y - 5) ** 2

            return loss1 * loss2

        self.pure_objective_function = booth_function

    def search_space(self, min=-10, max=10, value_types="array", size=10000):
        return super().create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
