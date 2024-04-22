# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
import pandas as pd
from functools import reduce

from search_data_collector import SqlSearchData
from hyperactive import Hyperactive
from hyperactive.optimizers import GridSearchOptimizer as HyperactiveGridSearchOptimizer
from gradient_free_optimizers import GridSearchOptimizer

from .config import default_search_data_path


class SurfacesDataCollector(SqlSearchData):
    def __init__(self, path=None) -> None:
        if path is None:
            path = default_search_data_path
        super().__init__(path, func2str=True)

    def _init_search_data(self, objective_function, search_space):
        self.para_names = [key for key in list(search_space.keys())]
        self.search_data_length = 0

        dim_sizes_list = [len(array) for array in search_space.values()]
        self.search_space_size = reduce((lambda x, y: x * y), dim_sizes_list)

        search_data_cols = self.para_names + ["score"]
        self.search_data = pd.DataFrame([], columns=search_data_cols)

    def _array_search_space(self, objective_function, search_space):
        while self.search_data_length < self.search_space_size:
            print("\n ------------ search_space_size", self.search_space_size)
            opt = GridSearchOptimizer(
                search_space,
                direction="orthogonal",
                initialize={},
            )
            opt.search(
                objective_function,
                n_iter=int(self.search_space_size * 1),
                verbosity=["progress_bar"],
            )

            self.search_data = pd.concat(
                [self.search_data, opt.search_data],
                ignore_index=True,
            )

            self.search_data = self.search_data.drop_duplicates(subset=self.para_names)
            self.search_data_length = len(self.search_data)
            print(
                "\n ------------ self.search_data_length", self.search_data_length, "\n"
            )

    def _list_search_space(self, objective_function, search_space):
        while self.search_data_length < self.search_space_size:
            print("\n ------------ search_space_size", self.search_space_size)

            hyper = Hyperactive(verbosity=["progress_bar"])
            hyper.add_search(
                objective_function,
                search_space,
                initialize={},
                n_iter=self.search_space_size,
                optimizer=HyperactiveGridSearchOptimizer(direction="orthogonal"),
                memory_warm_start=self.search_data,
            )
            hyper.run()

            self.search_data = pd.concat(
                [self.search_data, hyper.search_data(objective_function)],
                ignore_index=True,
            )

            self.search_data = self.search_data.drop_duplicates(subset=self.para_names)
            self.search_data_length = len(self.search_data)
            print(
                "\n ------------ self.search_data_length", self.search_data_length, "\n"
            )

    def collect(
        self,
        objective_function,
        search_space,
        table=None,
        if_exists="append",
    ):
        if table is None:
            table = objective_function.__name__

        self._init_search_data(objective_function, search_space)
        if isinstance(search_space[self.para_names[0]], np.ndarray):
            self._array_search_space(objective_function, search_space)
        else:
            self._list_search_space(objective_function, search_space)

        self.save(table, self.search_data, if_exists)

    def load(self, table):
        return self.sql_data.load(table)
