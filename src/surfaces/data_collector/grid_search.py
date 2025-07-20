import numpy as np
import itertools
from typing import Dict, List, Tuple, Union, Callable, Any


class GridSearchOptimizer:
    """
    A simple grid search optimizer that systematically evaluates all parameter combinations
    in a given search space. This implementation requires no external dependencies beyond numpy.
    """

    def __init__(
        self,
        search_space: Dict[str, Union[np.ndarray, List]],
        direction: str = "orthogonal",
    ):
        """
        Initialize the grid search optimizer.

        Parameters:
        -----------
        search_space : dict
            Dictionary mapping parameter names to their possible values (numpy arrays or lists)
        direction : str
            Grid traversal direction (kept for compatibility, not used in basic implementation)
        """
        self.search_space = search_space
        self.direction = direction
        self.para_names = list(search_space.keys())
        self.search_data = SearchData(self.para_names)

        # Calculate total search space size
        self.search_space_size = self._calculate_search_space_size()

    def _calculate_search_space_size(self) -> int:
        """Calculate the total number of parameter combinations."""
        size = 1
        for values in self.search_space.values():
            size *= len(values)
        return size

    def _generate_parameter_combinations(self) -> List[Tuple]:
        """Generate all parameter combinations using itertools.product."""
        param_values = [self.search_space[param] for param in self.para_names]
        return list(itertools.product(*param_values))

    def search(
        self,
        objective_function: Callable,
        n_iter: int = None,
        verbosity: List[str] = None,
        memory_warm_start: Any = None,
    ) -> None:
        """
        Execute the grid search.

        Parameters:
        -----------
        objective_function : callable
            Function to optimize, takes a dictionary of parameters
        n_iter : int
            Number of iterations (ignored for grid search, evaluates all combinations)
        verbosity : list
            List of verbosity options (e.g., ["progress_bar"])
        memory_warm_start : SearchData
            Previous search data to avoid re-evaluating points
        """
        show_progress = verbosity and "progress_bar" in verbosity

        # Get all parameter combinations
        all_combinations = self._generate_parameter_combinations()

        # Handle warm start - skip already evaluated combinations
        if memory_warm_start is not None and hasattr(memory_warm_start, "data"):
            existing_combos = set()
            for entry in memory_warm_start.data:
                combo = tuple(entry[param] for param in self.para_names)
                existing_combos.add(combo)

            # Filter out already evaluated combinations
            combinations_to_evaluate = [
                combo for combo in all_combinations if combo not in existing_combos
            ]

            # Copy existing data to our search_data
            for entry in memory_warm_start.data:
                self.search_data.add_entry(entry)
        else:
            combinations_to_evaluate = all_combinations

        # Evaluate each combination
        total = len(combinations_to_evaluate)
        for i, combo in enumerate(combinations_to_evaluate):
            # Create parameter dictionary
            para_dict = {param: value for param, value in zip(self.para_names, combo)}

            # Show progress if requested
            if show_progress and i % max(1, total // 100) == 0:
                progress = 100 * i / total
                print(f"Grid Search Progress: {i}/{total} ({progress:.1f}%)", end="\r")

            # Evaluate objective function
            try:
                score = objective_function(para_dict)
            except Exception as e:
                print(f"\nError evaluating {para_dict}: {e}")
                score = float("-inf")

            # Store result
            result = {**para_dict, "score": score}
            self.search_data.add_entry(result)

        if show_progress:
            print(f"Grid Search Progress: {total}/{total} (100.0%)")


class SearchData:
    """
    A simple data structure to store search results without pandas dependency.
    Mimics the interface of a pandas DataFrame for compatibility.
    """

    def __init__(self, para_names: List[str]):
        """
        Initialize the search data structure.

        Parameters:
        -----------
        para_names : list
            List of parameter names
        """
        self.para_names = para_names
        self.columns = para_names + ["score"]
        self.data = []
        self._index_map = {}  # For fast duplicate checking

    def add_entry(self, entry: Dict[str, Any]) -> None:
        """Add a new entry to the search data."""
        # Create a tuple key for duplicate checking
        key = tuple(entry[param] for param in self.para_names)

        # Only add if not a duplicate
        if key not in self._index_map:
            self._index_map[key] = len(self.data)
            self.data.append(entry)

    def drop_duplicates(self, subset: List[str] = None) -> "SearchData":
        """Remove duplicate entries based on subset of columns."""
        # Already handled in add_entry, but kept for compatibility
        return self

    def iterrows(self):
        """Iterate over rows, mimicking pandas interface."""
        for i, row in enumerate(self.data):
            yield i, row

    def __len__(self):
        """Return the number of entries."""
        return len(self.data)

    def __getitem__(self, key):
        """Get column data or specific entry."""
        if isinstance(key, str) and key in self.columns:
            # Return column as list
            return [entry.get(key) for entry in self.data]
        elif isinstance(key, int):
            # Return specific row
            return self.data[key]
        else:
            raise KeyError(f"Key {key} not found")

    def to_dict(self, orient: str = "records") -> Union[List[Dict], Dict[str, List]]:
        """Convert to dictionary format."""
        if orient == "records":
            return self.data.copy()
        elif orient == "list":
            result = {col: [] for col in self.columns}
            for entry in self.data:
                for col in self.columns:
                    result[col].append(entry.get(col))
            return result
        else:
            raise ValueError(f"Unsupported orient: {orient}")

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        if not self.data:
            return np.array([])

        # Convert to 2D array
        array_data = []
        for entry in self.data:
            row = [entry.get(col) for col in self.columns]
            array_data.append(row)

        return np.array(array_data)


class DataFrame:
    """
    A minimal DataFrame replacement for basic operations needed by SurfacesDataCollector.
    """

    def __init__(self, data=None, columns=None):
        """Initialize a minimal DataFrame."""
        if columns is None:
            columns = []

        self.columns = list(columns)
        self.data = []

        if data is not None:
            if isinstance(data, list):
                if data and isinstance(data[0], dict):
                    # List of dictionaries
                    self.data = data.copy()
                    if not self.columns and data:
                        self.columns = list(data[0].keys())
                elif data and isinstance(data[0], (list, tuple)):
                    # List of lists/tuples
                    for row in data:
                        entry = {col: val for col, val in zip(self.columns, row)}
                        self.data.append(entry)
            elif isinstance(data, dict):
                # Dictionary of lists
                num_rows = len(next(iter(data.values())))
                self.columns = list(data.keys())
                for i in range(num_rows):
                    entry = {col: data[col][i] for col in self.columns}
                    self.data.append(entry)

    def __len__(self):
        return len(self.data)

    def drop_duplicates(self, subset=None):
        """Remove duplicate rows based on subset of columns."""
        if subset is None:
            subset = self.columns

        seen = set()
        unique_data = []

        for entry in self.data:
            key = tuple(entry.get(col) for col in subset)
            if key not in seen:
                seen.add(key)
                unique_data.append(entry)

        new_df = DataFrame(columns=self.columns)
        new_df.data = unique_data
        return new_df

    def iterrows(self):
        """Iterate over rows."""
        for i, row in enumerate(self.data):
            yield i, row

    @staticmethod
    def concat(dfs, ignore_index=True):
        """Concatenate DataFrames."""
        if not dfs:
            return DataFrame()

        # Get columns from first non-empty DataFrame
        columns = None
        for df in dfs:
            if hasattr(df, "columns") and df.columns:
                columns = df.columns
                break

        if columns is None:
            return DataFrame()

        result = DataFrame(columns=columns)

        for df in dfs:
            if hasattr(df, "data"):
                result.data.extend(df.data)
            elif hasattr(df, "iterrows"):
                for _, row in df.iterrows():
                    result.data.append(row)

        return result
