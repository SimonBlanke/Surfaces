# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import time


class ObjectiveFunction:
    def __init__(self, metric="score", input_type="dictionary", sleep=0):
        self.metric = metric
        self.input_type = input_type
        self.sleep = sleep

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

    def __call__(self, *input):
        time.sleep(self.sleep)

        if self.input_type == "dictionary":
            return self.objective_function_dict(*input)
        elif self.input_type == "arrays":
            return self.objective_function_np(*input)
