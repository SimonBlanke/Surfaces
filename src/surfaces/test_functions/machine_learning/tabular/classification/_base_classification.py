# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .._base_tabular import BaseTabular


class BaseClassification(BaseTabular):
    def __init__(self, *args, metric="accuracy", **kwargs):
        super().__init__(*args, metric, **kwargs)
