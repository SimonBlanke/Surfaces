# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
from functools import reduce

from search_data_collector import SqlSearchData
from .grid_search import GridSearchOptimizer, DataFrame, SearchData
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

    def _perform_grid_search(self, objective_function, search_space):
        """
        Perform grid search using our custom implementation.
        This method handles both array and list search spaces uniformly.
        """
        while self.search_data_length < self.search_space_size:
            # Create optimizer with current search space
            opt = GridSearchOptimizer(
                search_space,
                direction="orthogonal",
            )

            # Create memory warm start if we have existing data
            memory_warm_start = None
            if len(self.search_data) > 0:
                memory_warm_start = SearchData(self.para_names)
                for _, row in self.search_data.iterrows():
                    memory_warm_start.add_entry(row)

            # Run the search
            opt.search(
                objective_function,
                n_iter=int(self.search_space_size * 1),
                verbosity=["progress_bar"],
                memory_warm_start=memory_warm_start,
            )

            # Convert search data to DataFrame format
            opt_df = DataFrame(opt.search_data.data, columns=opt.search_data.columns)

            # Concatenate with existing data
            self.search_data = DataFrame.concat(
                [self.search_data, opt_df],
                ignore_index=True,
            )

            # Remove duplicates
            self.search_data = self.search_data.drop_duplicates(subset=self.para_names)
            self.search_data_length = len(self.search_data)

    def _array_search_space(self, objective_function, search_space):
        while self.search_data_length < self.search_space_size:
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

    def _list_search_space(self, objective_function, search_space):
        while self.search_data_length < self.search_space_size:

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

    def collect(
        self,
        objective_function,
        search_space,
        table=None,
        if_exists="append",
    ):
        """
        Collect surface data by performing grid search on the objective function.

        Parameters:
        -----------
        objective_function : callable
            Function to optimize
        search_space : dict
            Dictionary mapping parameter names to their possible values
        table : str
            Name of the table to save results to
        if_exists : str
            How to handle existing data ('append', 'replace', etc.)
        """
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
