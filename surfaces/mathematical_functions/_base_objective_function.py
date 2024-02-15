# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
import pandas as pd
from functools import reduce

from gradient_free_optimizers import GridSearchOptimizer

from .._base_test_function import BaseTestFunction


class MathematicalFunction(BaseTestFunction):
    explanation = """ """

    dimensions = " "
    formula = r" "
    global_minimum = r" "

    def __init__(self, metric="score", input_type="dictionary", sleep=0):
        super().__init__()

        self.metric = metric
        self.input_type = input_type
        self.sleep = sleep

    @staticmethod
    def conv_arrays2lists(search_space):
        search_space_lists = {}
        for para_name, dim_values in search_space.items():
            search_space_lists[para_name] = list(dim_values)
        return search_space_lists

    def create_n_dim_search_space(self, min=-5, max=5, step=0.1, value_types="array"):
        search_space_ = {}

        for dim in range(self.n_dim):
            dim_str = "x" + str(dim)

            values = np.arange(min, max, step)
            if value_types == "list":
                values = list(values)
            search_space_[dim_str] = values

        return search_space_

    @staticmethod
    def search_space(search_space_blank, value_types):
        search_space = {}
        for para_names, blank_values in search_space_blank.items():
            print("\n blank_values \n", blank_values, "\n")
            dim_values = np.arange(*blank_values)
            if value_types == "list":
                dim_values = list(dim_values)
            search_space[para_names] = dim_values
        return search_space

    def search_data(self):
        self.search_space = self.search_space(value_types="list")

        para_names = list(self.search_space.keys())
        search_data_cols = para_names + ["score"]
        search_data = pd.DataFrame([], columns=search_data_cols)
        search_data_length = 0

        dim_sizes_list = [len(array) for array in self.search_space.values()]
        search_space_size = reduce((lambda x, y: x * y), dim_sizes_list)

        while search_data_length < search_space_size:
            print("\n ------------ search_space_size", search_space_size)
            opt = GridSearchOptimizer(
                self.search_space, direction="orthogonal", initialize={}
            )
            opt.search(self.objective_function_dict, n_iter=int(search_space_size * 1))

            search_data = pd.concat(
                [search_data, opt.search_data],
                ignore_index=True,
            )

            search_data = search_data.drop_duplicates(subset=para_names)
            search_data_length = len(search_data)
            print("\n ------------ search_data_length", search_data_length, "\n")

        return search_data

    def collect_data(self, table=None, if_exists="append"):
        if table is None:
            table = self.__name__
        self.sql_data.save(table, self.search_data(), if_exists)

    def return_metric(self, loss):
        if self.metric == "score":
            return -loss
        elif self.metric == "loss":
            return loss

    def objective_function_np(self, *args):
        para = {}
        for i, arg in enumerate(args):
            dim_str = "x" + str(i)
            para[dim_str] = arg

        return self.objective_function_dict(para)
