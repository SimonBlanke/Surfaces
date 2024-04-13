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
    def __init__(self, metric=None, sleep=0, load_search_data=False):
        super().__init__()

        self.metric = metric
        self.sleep = sleep

        self.create_objective_function()

        if load_search_data:
            self.objective_function = self.objective_function_loaded
        else:
            self.objective_function = self.pure_objective_function

        # self.objective_function.__func__.__name__ = self.__name__

    def _collect_data(self, table=None, if_exists="append"):
        if table is None:
            table = self.__name__

        search_space = self.search_space()
        search_data_cols = self.para_names + ["score"]
        search_data = pd.DataFrame([], columns=search_data_cols)
        search_data_length = 0

        dim_sizes_list = [len(array) for array in search_space.values()]
        search_space_size = reduce((lambda x, y: x * y), dim_sizes_list)

        while search_data_length < search_space_size:
            print("\n ------------ search_space_size", search_space_size)

            hyper = Hyperactive(verbosity=["progress_bar"])
            hyper.add_search(
                self.objective_function,
                search_space,
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

            search_data = search_data.drop_duplicates(subset=self.para_names)
            search_data_length = len(search_data)
            print("\n ------------ search_data_length", search_data_length, "\n")
        self.sql_data.save(self.__name__, search_data, if_exists)

    def objective_function_loaded(self, params):
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

        params_df = pd.DataFrame(parameter_d, index=[0])

        para_df_row = search_data[
            np.all(search_data[self.para_names].values == params_df.values, axis=1)
        ]
        score = para_df_row["score"].values[0]
        return score
