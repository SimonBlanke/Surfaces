# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


class ObjectiveFunction:
    def __init__(self, metric="score"):
        self.metric = metric

    def return_metric(self, loss):
        if self.metric == "score":
            return -loss
        elif self.metric == "loss":
            return loss
