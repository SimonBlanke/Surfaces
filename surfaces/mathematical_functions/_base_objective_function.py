# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import time
import numpy as np
import pandas as pd
from functools import reduce

from hyperactive import Hyperactive
from hyperactive.optimizers import GridSearchOptimizer

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

    def search_space(self, min=-5, max=5, step=0.1, value_types="array"):
        search_space_ = {}

        for dim in range(self.n_dim):
            dim_str = "x" + str(dim)

            values = np.arange(min, max, step)
            if value_types == "list":
                values = list(values)
            search_space_[dim_str] = values

        return search_space_

    def collect_data(self, if_exists="append"):
        self.search_space = self.search_space(value_types="list")

        para_names = list(self.search_space.keys())
        search_data_cols = para_names + ["score"]
        search_data = pd.DataFrame([], columns=search_data_cols)
        search_data_length = 0

        dim_sizes_list = [len(array) for array in self.search_space.values()]
        search_space_size = reduce((lambda x, y: x * y), dim_sizes_list)

        while search_data_length < search_space_size:
            hyper = Hyperactive(verbosity=["progress_bar"])
            hyper.add_search(
                self.objective_function_dict,
                self.search_space,
                initialize={},
                n_iter=search_space_size,
                optimizer=GridSearchOptimizer(direction="orthogonal"),
                memory_warm_start=search_data,
            )
            hyper.run()

            search_data = pd.concat(
                [search_data, hyper.search_data(self.objective_function_dict)],
                ignore_index=True,
            )

            search_data = search_data.drop_duplicates(subset=para_names)
            search_data_length = len(search_data)

        self.sql_data.save(self.__name__, search_data, if_exists)

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
