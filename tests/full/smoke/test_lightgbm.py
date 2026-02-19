import pytest

from surfaces.test_functions.machine_learning.hyperparameter_optimization.tabular import (
    LightGBMClassifierFunction,
    LightGBMRegressorFunction,
)


@pytest.mark.smoke
@pytest.mark.ml
def test_lightgbm_classifier_init():
    """Test that LightGBM Classifier instantiates and has a valid search space."""

    func = LightGBMClassifierFunction(dataset="digits", cv=2)
    func._create_objective_function()
    space = func.search_space
    config = {k: v[0] if isinstance(v, list) else v for k, v in space.items()}
    score = func.pure_objective_function(config)

    # verify output
    assert func is not None
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


@pytest.mark.smoke
@pytest.mark.ml
def test_lightgbm_regressor_init():
    """Test that LightGBM regressor instantiates and has a valid search space."""

    func = LightGBMRegressorFunction(dataset="diabetes", cv=2)
    func._create_objective_function()

    space = func.search_space
    config = {k: v[0] if isinstance(v, list) else v for k, v in space.items()}

    score = func.pure_objective_function(config)

    assert isinstance(score, float)
