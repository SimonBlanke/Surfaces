"""Formula regressions for multi-objective benchmark functions."""

import numpy as np
import pytest

from surfaces.test_functions.algebraic.multi_objective.dtlz.dtlz7 import (
    _PARETO_POSITION_INTERVALS,
    DTLZ7,
)
from surfaces.test_functions.algebraic.multi_objective.wfg import (
    WFG1,
    WFG2,
    WFG3,
    WFG4,
    WFG5,
    WFG6,
    WFG7,
    WFG8,
    WFG9,
)
from surfaces.test_functions.algebraic.multi_objective.wfg._transformations import s_linear
from surfaces.test_functions.algebraic.multi_objective.zdt.zdt3 import (
    _PARETO_SEGMENTS,
    ZDT3,
    _distribute_points,
)
from surfaces.test_functions.algebraic.multi_objective.zdt.zdt4 import ZDT4


def _in_any_interval(values, intervals):
    mask = np.zeros_like(values, dtype=bool)
    for lo, hi in intervals:
        mask |= (values >= lo - 1e-12) & (values <= hi + 1e-12)
    return mask


def _dtlz7_phi(values):
    return values * (1 + np.sin(3 * np.pi * values))


def test_wfg_s_linear_uses_asymmetric_scaling():
    values = np.array([0.0, 0.175, 0.35, 0.675, 1.0])
    expected = np.array([1.0, 0.5, 0.0, 0.5, 1.0])
    np.testing.assert_allclose(s_linear(values, 0.35), expected, atol=1e-12)


def test_wfg1_transition_pipeline_matches_reference_point():
    func = WFG1(n_objectives=2, n_dim=4)
    y = np.array([0.2, 0.7, 0.35, 0.9])

    np.testing.assert_allclose(
        func._transform(y),
        np.array([0.9847025179964968, 0.5688840415911236]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_wfg1_uses_mixed_shape_with_a5_alpha1():
    func = WFG1(n_objectives=2, n_dim=4)
    x = np.array([0.25, 0.0])
    expected = np.array(
        [
            2 * (1 - np.cos(x[0] * np.pi / 2)),
            4 * (1 - x[0] - np.cos(10 * np.pi * x[0] + np.pi / 2) / (10 * np.pi)),
        ]
    )

    np.testing.assert_allclose(func._compute_objectives(x), expected, atol=1e-12)


def test_wfg2_uses_disconnected_shape_with_a5_alpha1_beta1():
    func = WFG2(n_objectives=2, n_dim=4)
    x = np.array([0.25, 0.0])
    expected = np.array(
        [
            2 * (1 - np.cos(x[0] * np.pi / 2)),
            4 * (1 - x[0] * np.cos(5 * np.pi * x[0]) ** 2),
        ]
    )

    np.testing.assert_allclose(func._compute_objectives(x), expected, atol=1e-12)


@pytest.mark.parametrize("func_class", [WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9])
def test_wfg_pareto_set_zeroes_transformed_distance_component(func_class):
    func = func_class(n_objectives=3, n_dist=4)
    pset = func.pareto_set(n_points=8)

    distances = np.array([func._transform(func._normalize(x))[-1] for x in pset])
    np.testing.assert_allclose(distances, 0.0, atol=1e-10)


@pytest.mark.parametrize("func_class", [WFG2, WFG3])
def test_wfg_pairwise_nonseparable_distance_requires_even_length(func_class):
    with pytest.raises(ValueError, match="n_dist must be even"):
        func_class(n_objectives=2, n_dim=3)


def test_zdt3_distribution_never_returns_more_than_requested_points():
    for n_points in range(5):
        assert sum(_distribute_points(n_points)) == n_points
        assert ZDT3().pareto_front(n_points).shape == (n_points, 2)
        assert ZDT3().pareto_set(n_points).shape == (n_points, 30)


def test_zdt3_pareto_front_uses_reference_segments():
    func = ZDT3()
    front = func.pareto_front(n_points=37)

    assert np.all(_in_any_interval(front[:, 0], _PARETO_SEGMENTS))
    expected_f2 = 1 - np.sqrt(front[:, 0]) - front[:, 0] * np.sin(10 * np.pi * front[:, 0])
    np.testing.assert_allclose(front[:, 1], expected_f2, atol=1e-12)


def test_zdt4_exposes_nonuniform_variable_bounds():
    func = ZDT4(n_dim=4)

    assert func.spec.default_bounds == (-5.0, 5.0)
    assert func.variable_bounds == [(0.0, 1.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]

    search_space = func.search_space
    assert search_space["x0"].min() >= 0.0
    assert search_space["x0"].max() < 1.0
    for dim in range(1, func.n_dim):
        assert search_space[f"x{dim}"].min() >= -5.0
        assert search_space[f"x{dim}"].max() < 5.0


def test_dtlz7_pareto_front_samples_only_nondominated_intervals():
    func = DTLZ7(n_objectives=3)
    front = func.pareto_front(n_points=48)
    f_pos = front[:, :2]

    assert np.all(_in_any_interval(f_pos[:, 0], _PARETO_POSITION_INTERVALS))
    assert np.all(_in_any_interval(f_pos[:, 1], _PARETO_POSITION_INTERVALS))

    expected_last = 2 * func.n_objectives - np.sum(_dtlz7_phi(f_pos), axis=1)
    np.testing.assert_allclose(front[:, -1], expected_last, atol=1e-12)


def test_dtlz7_pareto_set_evaluates_to_the_sampled_front():
    func = DTLZ7(n_objectives=3)
    pset = func.pareto_set(n_points=32)

    assert np.all(pset[:, func.n_objectives - 1 :] == 0.0)
    assert np.all(_in_any_interval(pset[:, 0], _PARETO_POSITION_INTERVALS))
    assert np.all(_in_any_interval(pset[:, 1], _PARETO_POSITION_INTERVALS))

    np.testing.assert_allclose(func.batch(pset), func.pareto_front(n_points=32), atol=1e-12)
