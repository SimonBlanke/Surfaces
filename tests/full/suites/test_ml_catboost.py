import numpy as np
import pytest

from tests.conftest import get_sample_params


@pytest.mark.ml
def test_catboost_classifier(quick_ml_params):
    """CatBoostClassifier evaluates correctly."""
    from surfaces.test_functions.machine_learning import CatBoostClassifierFunction

    func = CatBoostClassifierFunction()
    params = {**get_sample_params(func), **quick_ml_params}
    result = func(params)

    assert isinstance(result, (int, float))
    assert np.isfinite(result)
