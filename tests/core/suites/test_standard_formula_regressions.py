import numpy as np
import pytest

from surfaces.test_functions.algebraic import (
    BoothFunction,
    BukinFunctionN6,
    EggholderFunction,
    LangermannFunction,
    LeviFunctionN13,
)


def assert_scalar_and_batch_match_reference(func, point, expected):
    point = np.asarray(point, dtype=float)

    scalar = func(point)
    batch = func.batch(point.reshape(1, -1))[0]

    np.testing.assert_allclose(scalar, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(batch, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.algebraic
@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((1.0, 1.0), 0.0),
        ((1.0, 0.25), 1.125),
        ((0.5, -0.5), 3.75),
    ],
)
def test_levi_n13_sfu_reference_values(point, expected):
    assert_scalar_and_batch_match_reference(LeviFunctionN13(), point, expected)


@pytest.mark.algebraic
@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((0.0, 0.0), -1.0271573538269159),
        ((2.0, 3.0), 2.3370763264942127),
        ((7.0, 9.0), 3.000037595874339),
    ],
)
def test_langermann_sfu_reference_values(point, expected):
    assert_scalar_and_batch_match_reference(LangermannFunction(), point, expected)


@pytest.mark.algebraic
def test_langermann_default_domain_matches_sfu_reference_domain():
    func = LangermannFunction()

    assert func.spec.default_bounds == (0.0, 10.0)

    for values in func.search_space.values():
        assert np.min(values) >= 0.0
        assert np.max(values) < 10.0
        assert np.max(values) >= 9.9


@pytest.mark.algebraic
@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((-10.0, 1.0), 0.0),
        ((-15.0, -3.0), 229.178784747792),
        ((-5.0, 3.0), 165.88123951777),
    ],
)
def test_bukin_n6_sfu_reference_values(point, expected):
    assert_scalar_and_batch_match_reference(BukinFunctionN6(), point, expected)


@pytest.mark.algebraic
def test_bukin_n6_default_search_space_matches_sfu_reference_domain():
    func = BukinFunctionN6()
    space = func.search_space

    assert np.isclose(np.min(space["x0"]), -15.0)
    assert np.max(space["x0"]) < -5.0
    assert np.max(space["x0"]) > -5.2
    assert np.isclose(np.min(space["x1"]), -3.0)
    assert np.max(space["x1"]) < 3.0
    assert np.max(space["x1"]) > 2.9

    assert np.min(space["x0"]) <= func.x_global[0] <= np.max(space["x0"])
    assert np.min(space["x1"]) <= func.x_global[1] <= np.max(space["x1"])


@pytest.mark.algebraic
def test_eggholder_default_search_space_matches_display_domain():
    func = EggholderFunction()
    space = func.search_space

    assert func.spec.default_bounds == (-512.0, 512.0)
    assert np.isclose(np.min(space["x0"]), -512.0)
    assert np.max(space["x0"]) < 512.0
    assert np.max(space["x0"]) > 500.0
    assert np.isclose(np.min(space["x1"]), -512.0)
    assert np.max(space["x1"]) < 512.0
    assert np.max(space["x1"]) > 500.0


@pytest.mark.algebraic
def test_booth_function_is_declared_convex():
    assert BoothFunction().spec.convex is True
