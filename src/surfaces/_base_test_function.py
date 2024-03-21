# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time

from .data_collector import SurfacesDataCollector


class BaseTestFunction:
    explanation = """ """

    dimensions = " "
    formula = r" "
    global_minimum = r" "

    pure_objective_function: callable

    def __init__(self):
        self.sql_data = SurfacesDataCollector()

        self.create_objective_function()

    def create_objective_function(self):
        e_msg = "'create_objective_function'-method is not implemented"
        raise NotImplementedError(e_msg)

    def search_space(self):
        e_msg = "'search_space'-method is not implemented"
        raise NotImplementedError(e_msg)

    def load_search_data(self):
        return self.sql_data.load(self.__name__)

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
