# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
import pandas as pd
from functools import reduce

from hyperactive import Hyperactive
from hyperactive.optimizers import GridSearchOptimizer
from .._base_test_function import BaseTestFunction


class MachineLearningFunction(BaseTestFunction):
    def __init__(self):
        super().__init__()

        self.objective_function.__func__.__name__ = self.__name__

    def collect_data(self, if_exists="append"):
        para_names = list(self.search_space.keys())
        search_data_cols = para_names + ["score"]
        search_data = pd.DataFrame([], columns=search_data_cols)
        search_data_length = 0

        dim_sizes_list = [len(array) for array in self.search_space.values()]
        search_space_size = reduce((lambda x, y: x * y), dim_sizes_list)

        while search_data_length < search_space_size:
            print("\n ------------ search_space_size", search_space_size)

            hyper = Hyperactive(verbosity=["progress_bar"])
            hyper.add_search(
                self.objective_function,
                self.search_space,
                initialize={},
                n_iter=search_space_size,
                optimizer=GridSearchOptimizer(direction="orthogonal"),
                memory_warm_start=search_data,
            )
            hyper.run()

            search_data = pd.concat(
                [search_data, hyper.search_data(self.objective_function)],
                ignore_index=True,
            )

            search_data = search_data.drop_duplicates(subset=para_names)
            search_data_length = len(search_data)
            print("\n ------------ search_data_length", search_data_length, "\n")
        self.sql_data.save(self.__name__, search_data, if_exists)

    def objective_function_dict(self, params):
        try:
            parameter_d = params.para_dict
        except AttributeError:
            parameter_d = params

        for para_names, dim_value in parameter_d.items():
            try:
                parameter_d[para_names] = dim_value.__name__
            except AttributeError:
                pass

        search_data = self.load_search_data()
        if search_data is None:
            msg = "Search Data is empty"
            raise TypeError(msg)
        para_names = list(self.search_space.keys())

        params_df = pd.DataFrame(parameter_d, index=[0])
        para_df_row = search_data[
            np.all(search_data[para_names].values == params_df.values, axis=1)
        ]

        score = para_df_row["score"].values[0]
        return score
