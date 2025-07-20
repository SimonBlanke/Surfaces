import os
from typing import Dict, Any, List


class SearchDataLookup:
    """
    Provides fast lookup of pre-computed objective function values.

    This class loads search data from disk and provides O(1) lookup
    for parameter combinations that were evaluated during grid search.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._load_data()
        self._build_lookup_table()

    def _load_data(self):
        """Load search data from file."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Search data file not found: {self.filepath}")

        data = np.load(self.filepath)
        self.parameters = data["parameters"]
        self.scores = data["scores"]
        self.times = data["times"]
        self.param_names = data["param_names"]

        # Reconstruct search space structure
        self.search_space_keys = data["search_space_keys"]
        self.search_space_sizes = data["search_space_sizes"]

    def _build_lookup_table(self):
        """Build hash table for fast parameter lookup."""
        self.lookup_table = {}

        for i, params in enumerate(self.parameters):
            # Create hashable key from parameters
            key = tuple(params)
            self.lookup_table[key] = {"score": self.scores[i], "time": self.times[i]}

    def evaluate(self, param_dict: Dict[str, Any]) -> float:
        """
        Look up objective function value for given parameters.

        Args:
            param_dict: Dictionary of parameter names to values

        Returns:
            Objective function value

        Raises:
            KeyError: If parameter combination not found in search data
        """
        # Convert dict to tuple in correct order
        param_values = [param_dict[name] for name in self.param_names]
        key = tuple(param_values)

        if key not in self.lookup_table:
            raise KeyError(
                f"Parameter combination not found in search data: {param_dict}"
            )

        return self.lookup_table[key]["score"]

    def get_evaluation_time(self, param_dict: Dict[str, Any]) -> float:
        """Get the original evaluation time for given parameters."""
        param_values = [param_dict[name] for name in self.param_names]
        key = tuple(param_values)

        if key not in self.lookup_table:
            raise KeyError(
                f"Parameter combination not found in search data: {param_dict}"
            )

        return self.lookup_table[key]["time"]
