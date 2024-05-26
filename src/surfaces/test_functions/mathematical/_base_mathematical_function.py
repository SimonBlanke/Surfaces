# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_test_function import BaseTestFunction


class MathematicalFunction(BaseTestFunction):
    explanation = """ """

    dimensions = " "
    formula = r" "
    global_minimum = r" "

    def __init__(
        self,
        metric="loss",
        sleep=0,
    ):
        super().__init__(metric, sleep)

        self.metric = metric
        self.sleep = sleep

        self._objective_function_ = self.pure_objective_function

    def return_metric(self, loss):
        if self.metric == "score":
            return -loss
        elif self.metric == "loss":
            return loss
        else:
            raise ValueError

    @staticmethod
    def conv_arrays2lists(search_space):
        search_space_lists = {}
        for para_name, dim_values in search_space.items():
            search_space_lists[para_name] = list(dim_values)
        return search_space_lists

    def create_n_dim_search_space(self, min=-5, max=5, size=100, value_types="array"):
        search_space_ = {}
        dim_size = size ** (1 / self.n_dim)

        def add_dim(search_space_: dict, dim: int, min, max):
            dim_str = "x" + str(dim)
            step_size = (max - min) / dim_size
            values = np.arange(min, max, step_size)
            if value_types == "list":
                values = list(values)
            search_space_[dim_str] = values

        if isinstance(min, list) and isinstance(max, list):
            if len(min) != len(max) or len(min) != self.n_dim:
                raise ValueError

            for dim, (min_, max_) in enumerate(zip(min, max)):
                add_dim(search_space_, dim, min_, max_)
        else:
            for dim in range(self.n_dim):
                add_dim(search_space_, dim, min, max)

        return search_space_
