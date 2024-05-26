# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time


class BaseTestFunction:
    explanation = """ """

    dimensions = " "
    formula = r" "
    global_minimum = r" "

    objective_function: callable
    pure_objective_function: callable

    def __init__(self, metric, sleep):
        self.sleep = sleep
        self.metric = metric

        self.create_objective_function()

    def create_objective_function(self):
        e_msg = "'create_objective_function'-method is not implemented"
        raise NotImplementedError(e_msg)

    def search_space(self):
        e_msg = "'search_space'-method is not implemented"
        raise NotImplementedError(e_msg)

    def return_metric(self, metric):
        return metric

    def objective_function_np(self, *args):
        para = {}
        for i, arg in enumerate(args):
            dim_str = "x" + str(i)
            para[dim_str] = arg

        return self._objective_function_(para)

    def objective_function(self, *input):
        time.sleep(self.sleep)

        metric = self.pure_objective_function(*input)
        return self.return_metric(metric)
