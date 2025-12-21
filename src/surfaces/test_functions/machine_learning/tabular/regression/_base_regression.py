# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .._base_tabular import BaseTabular


class BaseRegression(BaseTabular):
    def __init__(self, scoring="r2", *args, **kwargs):
        super().__init__(*args, scoring=scoring, **kwargs)
