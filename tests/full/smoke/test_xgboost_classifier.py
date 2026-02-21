import pytest

from surfaces.test_functions.machine_learning.hyperparameter_optimization.tabular import (
    XGBoostClassifierFunction,
)


@pytest.mark.ml
class TestXGBoostClassifier:
    def test_instantiation(self):
        func = XGBoostClassifierFunction()
        assert hasattr(func, "search_space")

    def test_evaluation(self):
        func = XGBoostClassifierFunction()
        params = {name: (bounds[0] + bounds[1]) / 2 for name, bounds in func.search_space.items()}
        result = func(params)
        assert isinstance(result, (int, float))
