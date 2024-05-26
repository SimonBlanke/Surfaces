# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .._base_tabular import BaseTabular


class BaseClassification(BaseTabular):
    metric = "accuracy"  # called 'scoring' in sklearn

    def __init__(self, metric, sleep, evaluate_from_data):
        super().__init__(metric, sleep, evaluate_from_data)
