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
        input_type="dictionary",
        sleep=0,
        evaluate_from_data=False,
    ):
        super().__init__(metric, sleep, evaluate_from_data)

        self.metric = metric
        self.input_type = input_type
        self.sleep = sleep

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

    @staticmethod
    def search_space_from_blank(search_space_blank, value_types):
        search_space = {}
        for para_names, blank_values in search_space_blank.items():
            dim_values = np.arange(*blank_values)
            if value_types == "list":
                dim_values = list(dim_values)
            search_space[para_names] = dim_values
        return search_space
