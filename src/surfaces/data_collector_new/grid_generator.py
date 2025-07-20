import numpy as np
from itertools import product
from typing import Dict, List, Tuple, Any
import time
import os


class GridGenerator:
    """
    Generates parameter grids from search space definitions.

    This class takes a search space dictionary and creates all possible
    combinations of parameters for grid search evaluation.
    """

    def __init__(self, search_space: Dict[str, List[Any]]):
        self.search_space = search_space
        self.param_names = list(search_space.keys())
        self.param_values = [search_space[name] for name in self.param_names]

    def generate_grid(self) -> Tuple[np.ndarray, List[str]]:
        """
        Generate complete parameter grid.

        Returns:
            grid: numpy array of shape (n_combinations, n_params)
            param_names: list of parameter names
        """
        # Create all combinations
        combinations = list(product(*self.param_values))
        grid = np.array(combinations)

        return grid, self.param_names

    def get_grid_size(self) -> int:
        """Calculate total number of grid points."""
        return np.prod([len(values) for values in self.param_values])
