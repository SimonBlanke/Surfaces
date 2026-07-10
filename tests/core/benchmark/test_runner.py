"""Tests for benchmark trial execution, in particular direction handling.

The benchmark stack assumes minimization everywhere (best_so_far via
min(), ERT targets, adapter setup). ML test functions default to
objective="maximize", so the runner must force "minimize" at
instantiation time. Without that, optimizers hunt for the worst
hyperparameters and every downstream statistic is silently inverted.
"""

from __future__ import annotations

import numpy as np
import pytest

from surfaces.benchmark._runner import _instantiate_function, _run_ask_tell
from surfaces.test_functions._base_single_objective import (
    BaseSingleObjectiveTestFunction,
)
from surfaces.test_functions.algebraic import SphereFunction
from surfaces.test_functions.machine_learning._base_machine_learning import (
    MachineLearningFunction,
)


class AccuracyLikeFunction(MachineLearningFunction):
    """Minimal ML-family function: maximize by default, peak at x0=0.6."""

    para_names = ["x0"]
    x0_default = list(np.linspace(0.0, 1.0, 11))

    def _ml_objective(self, params):
        return 1.0 - abs(params["x0"] - 0.6)


class HardcodedMaximizeFunction(BaseSingleObjectiveTestFunction):
    """Function whose direction cannot be configured by the runner."""

    def __init__(self, collect_data=True):
        super().__init__(objective="maximize", collect_data=collect_data)
        self.n_dim = 1

    def _objective(self, params):
        return params["x0"] ** 2

    def _default_search_space(self):
        return {"x0": np.linspace(-1.0, 1.0, 5)}


class SweepAdapter:
    """Ask/tell adapter that sweeps the x0 grid in order."""

    is_sealed = False

    def setup(self, search_space, seed, budget):
        self._values = list(search_space["x0"])
        self._index = 0

    def ask(self):
        value = self._values[self._index % len(self._values)]
        self._index += 1
        return {"x0": value}

    def tell(self, params, score):
        pass


class TestInstantiateFunction:
    def test_ml_function_forced_to_minimize(self):
        func = _instantiate_function(AccuracyLikeFunction)
        assert func.objective == "minimize"

    def test_ml_function_scores_are_minimize_oriented(self):
        func = _instantiate_function(AccuracyLikeFunction)
        good = func({"x0": 0.6})
        bad = func({"x0": 0.0})
        assert good < bad

    def test_algebraic_function_keeps_minimize(self):
        func = _instantiate_function(SphereFunction)
        assert func.objective == "minimize"
        assert func.collect_data is False

    def test_unforceable_maximize_function_is_rejected(self):
        with pytest.raises(ValueError, match="requires minimization"):
            _instantiate_function(HardcodedMaximizeFunction)


class TestRunAskTellDirection:
    def test_best_so_far_tracks_best_accuracy(self):
        func = _instantiate_function(AccuracyLikeFunction)
        trace = _run_ask_tell(func, SweepAdapter(), seed=0, budget_cu=None, budget_iter=11)

        assert len(trace) == 11
        # Peak accuracy 1.0 at x0=0.6 must surface as the best (lowest) score.
        assert trace.best_score == pytest.approx(-1.0)

        scores = [record.score for record in trace]
        running_min = np.minimum.accumulate(scores)
        assert [record.best_so_far for record in trace] == pytest.approx(list(running_min))
