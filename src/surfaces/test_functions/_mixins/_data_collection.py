# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Data collection mixin for tracking evaluation history."""

from typing import Any, Dict, Optional


class DataCollectionMixin:
    """Mixin providing evaluation data collection for test functions.

    Tracks evaluation history including search data, best score found,
    best parameters, number of evaluations, and total computation time.

    Attributes
    ----------
    n_evaluations : int
        Number of function evaluations performed.
    search_data : list of dict
        History of all evaluations as list of dicts containing parameters and score.
    best_score : float or None
        Best score found (respects objective direction).
    best_params : dict or None
        Parameters that achieved the best score.
    total_time : float
        Cumulative time spent in function evaluations (seconds).

    Notes
    -----
    This mixin expects the following attributes to be present:
    - self.objective: str ("minimize" or "maximize")
    - self.collect_data: bool
    - self._callbacks: list (from CallbackMixin)
    """

    n_evaluations: int
    search_data: list
    best_score: Optional[float]
    best_params: Optional[Dict[str, Any]]
    total_time: float

    def _init_data_collection(self) -> None:
        """Initialize data collection attributes."""
        self.n_evaluations = 0
        self.search_data = []
        self.best_score = None
        self.best_params = None
        self.total_time = 0.0

    def _record_evaluation(
        self,
        params: Dict[str, Any],
        score: float,
        elapsed_time: float = 0.0,
        from_cache: bool = False,
    ) -> None:
        """Record an evaluation and invoke callbacks.

        Parameters
        ----------
        params : dict
            The parameters that were evaluated.
        score : float
            The resulting score.
        elapsed_time : float, default=0.0
            Time taken for this evaluation in seconds.
        from_cache : bool, default=False
            Whether this result came from the memory cache.
        """
        record = {**params, "score": score}

        if self.collect_data:
            self.n_evaluations += 1
            self.search_data.append(record)

            # Update timing (only for non-cached evaluations)
            if not from_cache:
                self.total_time += elapsed_time

            # Update best score/params
            is_better = (
                self.best_score is None
                or (self.objective == "minimize" and score < self.best_score)
                or (self.objective == "maximize" and score > self.best_score)
            )
            if is_better:
                self.best_score = score
                self.best_params = params.copy()

        # Invoke callbacks
        for callback in self._callbacks:
            callback(record)

    def reset_data(self) -> None:
        """Reset all collected evaluation data.

        Clears n_evaluations, search_data, best_score, best_params, and total_time.
        Does not clear the memory cache (use reset_memory() for that).
        """
        self.n_evaluations = 0
        self.search_data = []
        self.best_score = None
        self.best_params = None
        self.total_time = 0.0
