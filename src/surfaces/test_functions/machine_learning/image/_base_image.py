# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .._base_machine_learning import MachineLearningFunction


class BaseImage(MachineLearningFunction):
    """Base class for image-based test functions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
