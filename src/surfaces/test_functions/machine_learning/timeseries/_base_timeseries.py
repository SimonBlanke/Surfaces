# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .._base_machine_learning import MachineLearningFunction


class BaseTimeSeries(MachineLearningFunction):
    """Base class for time-series test functions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
