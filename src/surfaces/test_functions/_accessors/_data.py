"""DataAccessor: proxy to evaluation data on BaseTestFunction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .._base_test_function import BaseTestFunction


class DataAccessor:
    """Namespaced access to evaluation data.

    Thin proxy over private attributes on BaseTestFunction.

    Parameters
    ----------
    func : BaseTestFunction
        The test function instance.
    """

    def __init__(self, func: "BaseTestFunction") -> None:
        self._func = func

    @property
    def n_evaluations(self) -> int:
        return self._func._n_evaluations

    @property
    def search_data(self) -> List[Dict[str, Any]]:
        return self._func._search_data

    @property
    def best_score(self) -> Optional[float]:
        return self._func._best_score

    @property
    def best_params(self) -> Optional[Dict[str, Any]]:
        return self._func._best_params

    @property
    def total_time(self) -> float:
        return self._func._total_time

    def reset(self) -> None:
        """Reset all collected evaluation data."""
        self._func._n_evaluations = 0
        self._func._search_data.clear()
        self._func._best_score = None
        self._func._best_params = None
        self._func._total_time = 0.0

    def __repr__(self) -> str:
        return f"DataAccessor(n_evaluations={self.n_evaluations}, " f"best_score={self.best_score})"
