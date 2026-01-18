import pytest

from surfaces.test_functions.machine_learning.tabular import (
    LightGBMClassifierFunction,
    LightGBMRegressorFunction,
)


@pytest.mark.smoke
@pytest.mark.ml
def test_lightgbm_classifier_init():
    """Test that LightGBM Classifier instantiates and has a valid search space."""
    try:
        func = LightGBMClassifierFunction()
    except ImportError:
        pytest.skip("LightGBM not installed")

    assert func is not None
    space = func.search_space()

    # key params
    assert "n_estimators" in space
    assert "learning_rate" in space
    assert "num_leaves" in space


@pytest.mark.smoke
@pytest.mark.ml
def test_lightgbm_regressor_init():
    """Test that LightGBM regressor instantiates and has a valid search space."""
    try:
        func = LightGBMRegressorFunction()
    except ImportError:
        pytest.skip("LightGBM not installed")
    assert func is not None
    space = func.search_space()
    # Key params
    assert "n_estimators" in space
    assert "learning_rate" in space
    assert "num_leaves" in space
