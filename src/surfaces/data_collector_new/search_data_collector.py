from typing import Dict, Any, List
import numpy as np
import time

from .grid_generator import GridGenerator


class SearchDataCollector:
    """
    Collects search data by evaluating objective functions on parameter grids.

    This class handles the expensive computation of evaluating ML models
    across a parameter grid and saves the results for future use.
    """

    def __init__(self, objective_function, search_space: Dict[str, List[Any]]):
        self.objective_function = objective_function
        self.search_space = search_space
        self.grid_generator = GridGenerator(search_space)

    def collect(self, verbose: bool = True) -> Dict[str, np.ndarray]:
        """
        Evaluate objective function on entire grid.

        Returns dictionary containing:
            - 'parameters': parameter grid
            - 'scores': objective function values
            - 'times': evaluation times in seconds
            - 'param_names': parameter names
        """
        grid, param_names = self.grid_generator.generate_grid()
        n_points = len(grid)

        scores = np.zeros(n_points)
        times = np.zeros(n_points)

        for i, params in enumerate(grid):
            if verbose and i % 100 == 0:
                print(f"Evaluating point {i+1}/{n_points}")

            # Convert to dictionary for objective function
            param_dict = {name: value for name, value in zip(param_names, params)}

            # Time the evaluation
            start_time = time.perf_counter()
            scores[i] = self.objective_function(param_dict)
            times[i] = time.perf_counter() - start_time

        return {
            "parameters": grid,
            "scores": scores,
            "times": times,
            "param_names": np.array(param_names, dtype="U"),  # Unicode string array
        }

    def save(self, filepath: str, verbose: bool = True):
        """Collect data and save to file."""
        data = self.collect(verbose=verbose)

        # Add metadata
        data["search_space_keys"] = np.array(list(self.search_space.keys()), dtype="U")
        data["search_space_sizes"] = np.array(
            [len(v) for v in self.search_space.values()]
        )

        # Save as compressed numpy archive
        np.savez_compressed(filepath, **data)

        if verbose:
            print(f"Saved search data to {filepath}")
            print(f"Total points: {len(data['scores'])}")
            print(f"Total evaluation time: {np.sum(data['times']):.2f} seconds")
