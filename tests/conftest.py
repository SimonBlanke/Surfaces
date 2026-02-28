"""Root pytest configuration for Surfaces test suite.

This file contains:
- Custom marker registration
- Auto-marker application based on test paths
- Shared helper functions for test utilities
"""

import pytest

# =============================================================================
# Helper Functions
# =============================================================================


def instantiate_function(func_class, n_dim=None):
    """Instantiate a test function with appropriate parameters.

    Uses the class's _spec['scalable'] attribute to determine if n_dim is required,
    rather than relying on try-except for control flow.
    """
    spec = getattr(func_class, "_spec", {})
    is_scalable = spec.get("scalable", False)

    if is_scalable or n_dim is not None:
        dim = n_dim if n_dim is not None else 2
        return func_class(n_dim=dim)
    return func_class()


def get_sample_params(func):
    """Get sample parameters from a function's search space."""
    search_space = func.search_space
    return {
        key: list(values)[0] if hasattr(values, "__iter__") else values
        for key, values in search_space.items()
    }


def get_middle_params(func):
    """Get middle-range parameters from a function's search space."""
    search_space = func.search_space
    params = {}
    for key, values in search_space.items():
        if hasattr(values, "__iter__"):
            values_list = list(values)
            mid_idx = len(values_list) // 2
            params[key] = values_list[mid_idx]
        else:
            params[key] = values
    return params


def func_id(func_class):
    """Generate a readable ID for a test function class."""
    return func_class.__name__


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def quick_ml_params():
    """Provide minimal ML parameters for quick classification testing.

    Uses small cv=2 to speed up cross-validation in tests.
    """
    from surfaces.test_functions.machine_learning.hyperparameter_optimization.tabular.classification.datasets import (
        iris_data,
    )

    return {
        "cv": 2,
        "dataset": iris_data,
    }


@pytest.fixture
def quick_regression_params():
    """Provide minimal ML parameters for quick regression testing.

    Uses small cv=2 to speed up cross-validation in tests.
    """
    from surfaces.test_functions.machine_learning.hyperparameter_optimization.tabular.regression.datasets import (
        diabetes_data,
    )

    return {
        "cv": 2,
        "dataset": diabetes_data,
    }


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "smoke: Quick sanity checks (<1s each)")
    config.addinivalue_line("markers", "slow: Tests that take >5s")
    config.addinivalue_line("markers", "ml: Machine learning functions (require sklearn)")
    config.addinivalue_line("markers", "cec: CEC competition functions")
    config.addinivalue_line("markers", "cec2013: CEC 2013 functions")
    config.addinivalue_line("markers", "cec2014: CEC 2014 functions")
    config.addinivalue_line("markers", "cec2017: CEC 2017 functions")
    config.addinivalue_line("markers", "bbob: BBOB benchmark functions")
    config.addinivalue_line("markers", "algebraic: Algebraic/mathematical functions")
    config.addinivalue_line("markers", "engineering: Engineering design functions")
    config.addinivalue_line("markers", "viz: Visualization tests (require plotly/matplotlib)")
    config.addinivalue_line("markers", "dashboard: Streamlit dashboard tests")
    # Interface compliance markers
    config.addinivalue_line("markers", "static: Tests that don't require instantiation (fast)")
    config.addinivalue_line("markers", "instance: Tests that require class instantiation")


def pytest_collection_modifyitems(config, items):
    """Auto-apply markers based on test location."""
    for item in items:
        test_path = str(item.fspath)

        if "test_ml" in test_path or "machine_learning" in test_path:
            item.add_marker(pytest.mark.ml)

        if "cec2013" in test_path:
            item.add_marker(pytest.mark.cec2013)
            item.add_marker(pytest.mark.cec)

        if "cec2014" in test_path:
            item.add_marker(pytest.mark.cec2014)
            item.add_marker(pytest.mark.cec)

        if "cec2017" in test_path:
            item.add_marker(pytest.mark.cec2017)
            item.add_marker(pytest.mark.cec)

        if "bbob" in test_path:
            item.add_marker(pytest.mark.bbob)

        if "algebraic" in test_path:
            item.add_marker(pytest.mark.algebraic)

        if "engineering" in test_path:
            item.add_marker(pytest.mark.engineering)

        if "smoke" in test_path:
            item.add_marker(pytest.mark.smoke)

        if "visualization" in test_path:
            item.add_marker(pytest.mark.viz)

        if "dashboard" in test_path:
            item.add_marker(pytest.mark.dashboard)

        if "integration" in test_path:
            item.add_marker(pytest.mark.slow)
