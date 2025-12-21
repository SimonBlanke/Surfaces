import math
import time

import pytest

from surfaces.test_functions import mathematical_functions

mathematical_functions_d = (
    "test_function",
    mathematical_functions,
)


@pytest.mark.parametrize(*mathematical_functions_d)
def test_memory_caching(test_function):
    """Test that memory caching stores and retrieves values correctly."""
    try:
        func = test_function(memory=True)
    except TypeError:
        func = test_function(n_dim=2, memory=True)

    search_space = func.search_space
    param_names = sorted(search_space.keys())
    params = {name: search_space[name][0] for name in param_names}

    result1 = func(params)
    result2 = func(params)

    # Handle NaN values (nan != nan, but both should be nan if cached correctly)
    if math.isnan(result1):
        assert math.isnan(result2)
    else:
        assert result1 == result2
    assert len(func._memory_cache) == 1


@pytest.mark.parametrize(*mathematical_functions_d)
def test_memory_cache_key(test_function):
    """Test that different positions create different cache entries."""
    try:
        func = test_function(memory=True)
    except TypeError:
        func = test_function(n_dim=2, memory=True)

    search_space = func.search_space
    param_names = sorted(search_space.keys())

    params1 = {name: search_space[name][0] for name in param_names}
    params2 = {name: search_space[name][-1] for name in param_names}

    func(params1)
    func(params2)

    assert len(func._memory_cache) == 2


@pytest.mark.parametrize(*mathematical_functions_d)
def test_memory_disabled_by_default(test_function):
    """Test that memory is disabled by default."""
    try:
        func = test_function()
    except TypeError:
        func = test_function(n_dim=2)

    assert func.memory is False
    assert len(func._memory_cache) == 0


def test_memory_skips_sleep_on_cache_hit():
    """Test that cached evaluations skip the sleep delay."""
    from surfaces.test_functions import SphereFunction

    func = SphereFunction(n_dim=2, memory=True, sleep=0.1)

    start = time.time()
    func([0.0, 0.0])
    first_call_time = time.time() - start

    start = time.time()
    func([0.0, 0.0])
    second_call_time = time.time() - start

    assert first_call_time >= 0.1
    assert second_call_time < 0.05
