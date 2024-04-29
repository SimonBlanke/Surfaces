# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time

import numpy as np
import pandas as pd

from ..data_collector import SurfacesDataCollector


class BaseTestFunction:
    explanation = """ """

    dimensions = " "
    formula = r" "
    global_minimum = r" "

    objective_function: callable
    pure_objective_function: callable

    def __init__(self, metric, sleep, evaluate_from_data):
        self.sleep = sleep
        self.metric = metric

        self.create_objective_function()

        if evaluate_from_data:
            self.sdc = SurfacesDataCollector()
            self.objective_function = self.objective_function_loaded
        else:
            self.objective_function = self.pure_objective_function

        self.objective_function.__name__ = self.__name__

    def create_objective_function(self):
        e_msg = "'create_objective_function'-method is not implemented"
        raise NotImplementedError(e_msg)

    def search_space(self):
        e_msg = "'search_space'-method is not implemented"
        raise NotImplementedError(e_msg)

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

        return self.pure_objective_function(para)

    def objective_function(self, *input):
        time.sleep(self.sleep)

        if self.input_type == "dictionary":
            metric = self.pure_objective_function(*input)
        elif self.input_type == "arrays":
            metric = self.objective_function_np(*input)

        return self.return_metric(metric)

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

        search_data = self.sdc.load(self.__name__)
        if search_data is None:
            msg = "Search Data is empty"
            raise TypeError(msg)

        params_df = pd.DataFrame(parameter_d, index=[0])

        para_df_row = search_data[
            np.all(search_data[self.para_names].values == params_df.values, axis=1)
        ]
        score = para_df_row["score"].values[0]
        return score
