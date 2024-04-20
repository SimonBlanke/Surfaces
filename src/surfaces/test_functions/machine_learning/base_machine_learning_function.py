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
