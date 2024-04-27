# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .._base_test_function import BaseTestFunction


class MachineLearningFunction(BaseTestFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
