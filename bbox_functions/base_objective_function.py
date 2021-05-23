# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


class ObjectiveFunction:
    def __init__(self, metric="score", parameter_type="dictionary"):
        self.metric = metric
        self.parameter_type = parameter_type

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
        if self.parameter_type == "dictionary":
            return self.objective_function_dict(*input)
        elif self.parameter_type == "arrays":
            return self.objective_function_np(*input)
