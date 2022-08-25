import os
import time
import numpy as np

from gradient_free_optimizers import GridSearchOptimizer
from simple_data_collector import DataCollector

from sklearn.neighbors import KNeighborsClassifier as KNeighborsClassifier_
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris


here_path = os.path.dirname(os.path.realpath(__file__))


class BaseMachineLearningFunction:
    __name__ = "k_neighbors_classifier"

    def __init__(self, X, y, cv=3, metric="score", input_type="dictionary", sleep=0):
        self.X = X
        self.y = y
        self.cv = cv
        self.metric = metric
        self.input_type = input_type
        self.sleep = sleep

        self.collector = DataCollector(
            os.path.join(here_path, "data", "search_data.csv")
        )

    def collect_data(self):
        opt = GridSearchOptimizer(self.search_space)

        search_data = []
        search_data_length = 0
        search_space_size = opt.conv.search_space_size

        while search_data_length < search_space_size:
            opt.search(
                self.model,
                n_iter=search_space_size,
                memory_warm_start=search_data,
                verbosity=["progress_bar"],
            )
            search_data = opt.search_data

            para_names = list(self.search_space.keys())
            search_data = search_data.drop_duplicates(subset=para_names)
            search_data_length = len(search_data)

        self.collector.save(search_data)

    def objective_function_dict(self, params):
        print("\n params \n", params, "\n")
        search_data = self.collector.load()
        print(len(search_data))

        params_l = list(search_data.columns)
        params_l.remove("score")

        for param in params_l:
            # search_data = search_data[search_data[param] == params[param]]

            search_data_values = search_data[param].values
            para_values = params[param]

        score = search_data["score"].values[0]
        print(" score", score, "\n")
        return score

    def __call__(self, *input):
        time.sleep(self.sleep)

        if self.input_type == "dictionary":
            return self.objective_function_dict(*input)
        elif self.input_type == "arrays":
            return self.objective_function_np(*input)
