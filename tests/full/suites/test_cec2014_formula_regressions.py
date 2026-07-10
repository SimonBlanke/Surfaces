"""Formula regressions for CEC 2014 hybrid and composition functions."""

import numpy as np
import pytest

import surfaces.test_functions.cec.cec2014.composition as cec2014_composition
import surfaces.test_functions.cec.cec2014.hybrid as cec2014_hybrid
from surfaces.test_functions.cec.cec2014 import (
    CompositionFunction1,
    CompositionFunction2,
    CompositionFunction3,
    CompositionFunction4,
    CompositionFunction5,
    CompositionFunction6,
    CompositionFunction7,
    CompositionFunction8,
    HybridFunction1,
    HybridFunction2,
    HybridFunction3,
    HybridFunction4,
    HybridFunction5,
    HybridFunction6,
)

HYBRID_SPECS = [
    (
        HybridFunction1,
        ("schwefel", "rastrigin", "high_conditioned_elliptic"),
        [0.3, 0.3, 0.4],
    ),
    (
        HybridFunction2,
        ("bent_cigar", "hgbat", "rastrigin"),
        [0.3, 0.3, 0.4],
    ),
    (
        HybridFunction3,
        ("griewank", "weierstrass", "rosenbrock", "expanded_scaffer"),
        [0.2, 0.2, 0.3, 0.3],
    ),
    (
        HybridFunction4,
        ("hgbat", "discus", "expanded_griewank_rosenbrock", "rastrigin"),
        [0.2, 0.2, 0.3, 0.3],
    ),
    (
        HybridFunction5,
        ("expanded_scaffer", "hgbat", "rosenbrock", "schwefel", "high_conditioned_elliptic"),
        [0.1, 0.2, 0.2, 0.2, 0.3],
    ),
    (
        HybridFunction6,
        ("katsuura", "happycat", "expanded_griewank_rosenbrock", "schwefel", "ackley"),
        [0.1, 0.2, 0.2, 0.2, 0.3],
    ),
]


COMPOSITION_SPECS = [
    (
        CompositionFunction1,
        (
            "rosenbrock",
            "high_conditioned_elliptic",
            "bent_cigar",
            "discus",
            "high_conditioned_elliptic",
        ),
        [10, 20, 30, 40, 50],
        [1, 1e-6, 1e-26, 1e-6, 1e-6],
    ),
    (
        CompositionFunction2,
        ("schwefel", "rastrigin", "hgbat"),
        [20, 20, 20],
        [1, 1, 1],
    ),
    (
        CompositionFunction3,
        ("schwefel", "rastrigin", "high_conditioned_elliptic"),
        [10, 30, 50],
        [0.25, 1, 1e-7],
    ),
    (
        CompositionFunction4,
        ("schwefel", "happycat", "high_conditioned_elliptic", "weierstrass", "griewank"),
        [10, 10, 10, 10, 10],
        [0.25, 1, 1e-7, 2.5, 10],
    ),
    (
        CompositionFunction5,
        ("hgbat", "rastrigin", "schwefel", "weierstrass", "high_conditioned_elliptic"),
        [10, 10, 10, 20, 20],
        [10, 10, 2.5, 25, 1e-6],
    ),
    (
        CompositionFunction6,
        (
            "expanded_griewank_rosenbrock",
            "happycat",
            "schwefel",
            "expanded_scaffer",
            "high_conditioned_elliptic",
        ),
        [10, 20, 30, 40, 50],
        [2.5, 10, 2.5, 5e-4, 1e-6],
    ),
    (
        CompositionFunction7,
        ("hybrid1", "hybrid2", "hybrid3"),
        [10, 30, 50],
        [1, 1, 1],
    ),
    (
        CompositionFunction8,
        ("hybrid4", "hybrid5", "hybrid6"),
        [10, 30, 50],
        [1, 1, 1],
    ),
]


@pytest.mark.parametrize(("func_class", "components", "proportions"), HYBRID_SPECS)
def test_cec2014_hybrid_components_match_report(func_class, components, proportions):
    func = func_class(n_dim=10)

    assert func.component_names == components
    assert func.proportions == proportions


@pytest.mark.parametrize(("func_class", "components", "sigmas", "lambdas"), COMPOSITION_SPECS)
def test_cec2014_composition_components_match_report(func_class, components, sigmas, lambdas):
    func = func_class(n_dim=10)

    assert func.component_names == components
    assert func.sigmas == sigmas
    assert func.lambdas == lambdas


@pytest.mark.parametrize("module", [cec2014_hybrid, cec2014_composition])
def test_cec2014_schwefel_helpers_use_precise_global_offset(module):
    z = np.zeros(10)

    assert module._schwefel(z) == pytest.approx(0.0, abs=1e-8)
    np.testing.assert_allclose(module._batch_schwefel(z.reshape(1, -1)), [0.0], atol=1e-8)


@pytest.mark.parametrize("func_class", [spec[0] for spec in HYBRID_SPECS])
def test_cec2014_hybrid_global_optimum_matches_report_bias(func_class):
    func = func_class(n_dim=10)

    scalar = func(func.x_global)
    batch = func.batch(np.asarray([func.x_global]))[0]

    assert scalar == pytest.approx(func.f_global, abs=1e-3)
    assert batch == pytest.approx(scalar, abs=1e-10)


@pytest.mark.parametrize("func_class", [spec[0] for spec in COMPOSITION_SPECS])
def test_cec2014_composition_weights_are_one_hot_at_component_optima(func_class):
    func = func_class(n_dim=10)
    optima = func._get_composition_optima()

    for component_idx in range(func.n_functions):
        expected = np.zeros(func.n_functions)
        expected[component_idx] = 1.0

        scalar_weights = func._compute_weights(optima[component_idx], optima)
        batch_weights = func._batch_compute_weights(
            np.asarray([optima[component_idx]]),
            optima,
        )[0]

        np.testing.assert_allclose(scalar_weights, expected, atol=0, rtol=0)
        np.testing.assert_allclose(batch_weights, expected, atol=0, rtol=0)


@pytest.mark.parametrize("func_class", [spec[0] for spec in COMPOSITION_SPECS])
def test_cec2014_composition_component_optima_match_report_biases(func_class):
    func = func_class(n_dim=10)
    optima = func._get_composition_optima()[: func.n_functions]
    expected = func.f_global + np.asarray(func.biases)

    scalar = np.asarray([func(optimum) for optimum in optima])
    batch = func.batch(optima)

    np.testing.assert_allclose(scalar, expected, atol=1e-3, rtol=0)
    np.testing.assert_allclose(batch, scalar, atol=1e-10, rtol=0)


@pytest.mark.parametrize("func_class", [spec[0] for spec in COMPOSITION_SPECS])
def test_cec2014_composition_batch_matches_scalar_around_component_optima(func_class):
    func = func_class(n_dim=10)
    optima = func._get_composition_optima()[: func.n_functions]
    perturbation = np.linspace(-0.05, 0.05, func.n_dim)
    points = np.vstack([optimum + perturbation for optimum in optima])

    scalar = np.asarray([func(point) for point in points])
    batch = func.batch(points)

    np.testing.assert_allclose(batch, scalar, rtol=1e-10, atol=1e-8)


@pytest.mark.parametrize("func_class", [spec[0] for spec in HYBRID_SPECS + COMPOSITION_SPECS])
def test_cec2014_array_scalar_input_uses_numeric_coordinate_order(func_class):
    func = func_class(n_dim=20)
    point = np.linspace(-3.0, 3.0, func.n_dim)

    scalar = func(point)
    batch = func.batch(np.asarray([point]))[0]

    assert scalar == pytest.approx(batch, rel=1e-12, abs=1e-8)
