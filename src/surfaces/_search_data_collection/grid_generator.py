# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from itertools import product
from typing import Any, Dict, Iterator, List


class GridGenerator:
    """
    Generates parameter grids for exhaustive search space exploration.

    This class creates all possible parameter combinations from a search space
    dictionary, enabling comprehensive evaluation of machine learning functions.
    """

    @staticmethod
    def generate_grid(search_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations from a search space dictionary.

        Args:
            search_space: Dictionary where keys are parameter names and values
                         are lists of possible parameter values

        Returns:
            List of dictionaries, each representing a parameter combination

        Example:
            >>> search_space = {
            ...     'n_estimators': [10, 50, 100],
            ...     'max_depth': [3, 5, 7]
            ... }
            >>> grid = GridGenerator.generate_grid(search_space)
            >>> len(grid)
            9
            >>> grid[0]
            {'n_estimators': 10, 'max_depth': 3}
        """
        if not search_space:
            return []

        # Get parameter names and their possible values
        param_names = list(search_space.keys())
        param_values = list(search_space.values())

        # Generate all combinations using itertools.product
        combinations = []
        for value_combination in product(*param_values):
            param_dict = dict(zip(param_names, value_combination))
            combinations.append(param_dict)

        return combinations

    @staticmethod
    def generate_grid_iterator(search_space: Dict[str, List[Any]]) -> Iterator[Dict[str, Any]]:
        """
        Generate parameter combinations as an iterator for memory efficiency.

        This is useful for large search spaces where storing all combinations
        in memory would be impractical.

        Args:
            search_space: Dictionary where keys are parameter names and values
                         are lists of possible parameter values

        Yields:
            Dictionary representing a parameter combination

        Example:
            >>> search_space = {'a': [1, 2], 'b': ['x', 'y']}
            >>> for params in GridGenerator.generate_grid_iterator(search_space):
            ...     print(params)
            {'a': 1, 'b': 'x'}
            {'a': 1, 'b': 'y'}
            {'a': 2, 'b': 'x'}
            {'a': 2, 'b': 'y'}
        """
        if not search_space:
            return

        param_names = list(search_space.keys())
        param_values = list(search_space.values())

        for value_combination in product(*param_values):
            yield dict(zip(param_names, value_combination))

    @staticmethod
    def count_combinations(search_space: Dict[str, List[Any]]) -> int:
        """
        Count the total number of parameter combinations without generating them.

        Args:
            search_space: Dictionary where keys are parameter names and values
                         are lists of possible parameter values

        Returns:
            Total number of parameter combinations

        Example:
            >>> search_space = {'a': [1, 2, 3], 'b': ['x', 'y']}
            >>> GridGenerator.count_combinations(search_space)
            6
        """
        if not search_space:
            return 0

        total = 1
        for param_values in search_space.values():
            total *= len(param_values)

        return total

    @staticmethod
    def validate_search_space(search_space: Dict[str, List[Any]]) -> List[str]:
        """
        Validate a search space dictionary and return any issues found.

        Args:
            search_space: Dictionary to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not isinstance(search_space, dict):
            errors.append("Search space must be a dictionary")
            return errors

        if not search_space:
            errors.append("Search space cannot be empty")
            return errors

        for param_name, param_values in search_space.items():
            if not isinstance(param_name, str):
                errors.append(f"Parameter name must be string, got {type(param_name)}")

            if not isinstance(param_values, list):
                errors.append(
                    f"Parameter values for '{param_name}' must be a list, got {type(param_values)}"
                )
                continue

            if len(param_values) == 0:
                errors.append(f"Parameter values for '{param_name}' cannot be empty")

        return errors

    @staticmethod
    def get_search_space_info(search_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Get information about a search space.

        Args:
            search_space: Search space dictionary

        Returns:
            Dictionary containing search space statistics
        """
        info = {
            "parameter_count": len(search_space),
            "parameters": list(search_space.keys()),
            "total_combinations": GridGenerator.count_combinations(search_space),
            "parameter_sizes": {},
        }

        for param_name, param_values in search_space.items():
            info["parameter_sizes"][param_name] = len(param_values)

        return info
