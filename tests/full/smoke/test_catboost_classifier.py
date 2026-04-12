import pytest

from surfaces.test_functions.machine_learning.hyperparameter_optimization.tabular import (
    CatBoostClassifierFunction,
)


@pytest.mark.smoke
@pytest.mark.ml
def test_catboost_classifier_init():
    """Test that CatBoost Classifier instantiates and has a valid search space."""

    func = CatBoostClassifierFunction(dataset="digits", cv=2)
    space = func.search_space
    config = {k: v[0] if isinstance(v, list) else v for k, v in space.items()}
    score = func._ml_objective(config)

    assert func is not None
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
