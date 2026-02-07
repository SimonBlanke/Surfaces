# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import Any

from .._base_pipeline import BasePipeline


class BaseTabularPipeline(BasePipeline):
    """Base class for tabular pipeline optimization test functions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
