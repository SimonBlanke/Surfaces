# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .._base_test_function import BaseTestFunction


class MachineLearningFunction(BaseTestFunction):
    def __init__(self, metric, sleep, evaluate_from_data):
        super().__init__(metric, sleep, evaluate_from_data)
