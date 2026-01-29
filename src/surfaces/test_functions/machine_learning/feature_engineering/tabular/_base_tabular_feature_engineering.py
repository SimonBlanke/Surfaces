# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import Any

from .._base_feature_engineering import BaseFeatureEngineering


class BaseTabularFeatureEngineering(BaseFeatureEngineering):
    """Base class for tabular feature engineering test functions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
