import os
import time
import numpy as np
import pandas as pd

from functools import reduce

from hyperactive import Hyperactive
from hyperactive.optimizers import GridSearchOptimizer
from search_data_collector import SearchDataCollector


here_path = os.path.dirname(os.path.realpath(__file__))


class BaseMachineLearningFunction:
    def __init__(self, X, y, cv=3, metric="score", input_type="dictionary", sleep=0):
        self.X = X
        self.y = y
        self.cv = cv
        self.metric = metric
        self.input_type = input_type
        self.sleep = sleep

        self.init_sdc()

    def init_sdc(self):
        self.collector = SearchDataCollector(
            os.path.join(here_path, "data", self.__name__ + ".csv")
        )

    def collect_data(self):
        para_names = list(self.search_space.keys())
        search_data_cols = para_names + ["score"]
        search_data = pd.DataFrame([], columns=search_data_cols)
        search_data_length = 0

        dim_sizes_list = [len(array) for array in self.search_space.values()]
        search_space_size = reduce((lambda x, y: x * y), dim_sizes_list)

        while search_data_length < search_space_size:
            hyper = Hyperactive(verbosity=["progress_bar"])
            hyper.add_search(
                self.model,
                self.search_space,
                n_iter=search_space_size,
                optimizer=GridSearchOptimizer(),
                memory_warm_start=search_data,
            )
            hyper.run()

            search_data = pd.concat(
                [search_data, hyper.search_data(self.model)], ignore_index=True
            )
            search_data = search_data.drop_duplicates(subset=para_names)
            search_data_length = len(search_data)
            print("search_data_length", search_data_length)

        self.collector.save(search_data)

    def load_search_data(self):
        return self.collector.load()

    def objective_function_dict(self, params):
        try:
            parameter_d = params.para_dict
        except AttributeError:
            parameter_d = params

        search_data = self.load_search_data()
        para_names = list(self.search_space.keys())

        params_df = pd.DataFrame(parameter_d, index=[0])
        para_df_row = search_data[
            np.all(search_data[para_names].values == params_df.values, axis=1)
        ]

        score = para_df_row["score"].values[0]
        return score

    def __call__(self, *input):
        time.sleep(self.sleep)

        if self.input_type == "dictionary":
            return self.objective_function_dict(*input)
        elif self.input_type == "arrays":
            return self.objective_function_np(*input)
