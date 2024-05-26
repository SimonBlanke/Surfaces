# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .._base_machine_learning import MachineLearningFunction


class BaseTabular(MachineLearningFunction):
    def __init__(self, metric, sleep, evaluate_from_data):
        super().__init__(metric, sleep, evaluate_from_data)
