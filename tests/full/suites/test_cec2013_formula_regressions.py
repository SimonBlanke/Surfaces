"""Report-based formula regressions for CEC 2013."""

import numpy as np
import pytest

from surfaces.test_functions.cec.cec2013 import (
    CompositionFunction1,
    CompositionFunction2,
    CompositionFunction3,
    CompositionFunction4,
    CompositionFunction5,
    CompositionFunction6,
    CompositionFunction7,
    CompositionFunction8,
    DifferentPowers,
    LunacekBiRastrigin,
    Rastrigin,
    RotatedAckley,
    RotatedBentCigar,
    RotatedDiscus,
    RotatedExpandedGriewankRosenbrock,
    RotatedExpandedScafferF6,
    RotatedGriewank,
    RotatedHighConditionedElliptic,
    RotatedKatsuura,
    RotatedLunacekBiRastrigin,
    RotatedRastrigin,
    RotatedRosenbrock,
    RotatedSchafferF7,
    RotatedSchwefel,
    RotatedWeierstrass,
    Schwefel,
    Sphere,
    StepRastrigin,
)

NON_COMPOSITION_FUNCTIONS = [
    Sphere,
    RotatedHighConditionedElliptic,
    RotatedBentCigar,
    RotatedDiscus,
    DifferentPowers,
    RotatedRosenbrock,
    RotatedSchafferF7,
    RotatedAckley,
    RotatedWeierstrass,
    RotatedGriewank,
    Rastrigin,
    RotatedRastrigin,
    StepRastrigin,
    Schwefel,
    RotatedSchwefel,
    RotatedKatsuura,
    LunacekBiRastrigin,
    RotatedLunacekBiRastrigin,
    RotatedExpandedGriewankRosenbrock,
    RotatedExpandedScafferF6,
]


COMPOSITION_CASES = [
    (
        CompositionFunction1,
        [10, 20, 30, 40, 50],
        [1, 1e-6, 1e-26, 1e-6, 0.1],
        ["rosenbrock", "different_powers", "bent_cigar", "discus", "sphere"],
    ),
    (CompositionFunction2, [20, 20, 20], [1, 1, 1], ["schwefel", "schwefel", "schwefel"]),
    (
        CompositionFunction3,
        [20, 20, 20],
        [1, 1, 1],
        ["rotated_schwefel", "rotated_schwefel", "rotated_schwefel"],
    ),
    (
        CompositionFunction4,
        [20, 20, 20],
        [0.25, 1, 2.5],
        ["rotated_schwefel", "rotated_rastrigin", "weierstrass"],
    ),
    (
        CompositionFunction5,
        [10, 30, 50],
        [0.25, 1, 2.5],
        ["rotated_schwefel", "rotated_rastrigin", "weierstrass"],
    ),
    (
        CompositionFunction6,
        [10, 10, 10, 10, 10],
        [0.25, 1, 1e-7, 2.5, 10],
        ["rotated_schwefel", "rotated_rastrigin", "elliptic", "weierstrass", "griewank"],
    ),
    (
        CompositionFunction7,
        [10, 10, 10, 20, 20],
        [100, 10, 2.5, 25, 0.1],
        ["griewank", "rotated_rastrigin", "rotated_schwefel", "weierstrass", "sphere"],
    ),
    (
        CompositionFunction8,
        [10, 20, 30, 40, 50],
        [2.5, 2.5e-3, 2.5, 5e-4, 0.1],
        [
            "expanded_griewank_rosenbrock",
            "schaffer_f7",
            "rotated_schwefel",
            "expanded_schaffer_f6",
            "sphere",
        ],
    ),
]


REPORT_F_GLOBAL = {
    1: -1400,
    2: -1300,
    3: -1200,
    4: -1100,
    5: -1000,
    6: -900,
    7: -800,
    8: -700,
    9: -600,
    10: -500,
    11: -400,
    12: -300,
    13: -200,
    14: -100,
    15: 100,
    16: 200,
    17: 300,
    18: 400,
    19: 500,
    20: 600,
    21: 700,
    22: 800,
    23: 900,
    24: 1000,
    25: 1100,
    26: 1200,
    27: 1300,
    28: 1400,
}


def test_cec2013_array_input_uses_numeric_coordinate_order_for_n_dim_20():
    func = Sphere(n_dim=20)
    x = np.linspace(-3.0, 4.0, func.n_dim)
    params = {f"x{i}": x[i] for i in range(func.n_dim)}

    assert func(x) == pytest.approx(func(params), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    "func_class", NON_COMPOSITION_FUNCTIONS + [case[0] for case in COMPOSITION_CASES]
)
def test_report_global_bias_table(func_class):
    func = func_class(n_dim=10)
    assert func.f_global == REPORT_F_GLOBAL[func.func_id]


@pytest.mark.parametrize("func_class", NON_COMPOSITION_FUNCTIONS)
def test_non_composition_optimum_matches_report_bias_scalar_and_batch(func_class):
    func = func_class(n_dim=10)
    expected = REPORT_F_GLOBAL[func.func_id]

    assert func(func.x_global) == pytest.approx(expected, abs=1e-8)
    np.testing.assert_allclose(func.batch(np.asarray([func.x_global])), [expected], atol=1e-8)


@pytest.mark.parametrize("func_class, sigmas, lambdas, components", COMPOSITION_CASES)
def test_composition_report_tables(func_class, sigmas, lambdas, components):
    func = func_class(n_dim=10)

    assert func.sigmas == sigmas
    assert func.lambdas == pytest.approx(lambdas)
    assert func.component_functions == components


@pytest.mark.parametrize("func_class, sigmas, lambdas, components", COMPOSITION_CASES)
def test_composition_component_optima_are_report_biases(func_class, sigmas, lambdas, components):
    func = func_class(n_dim=10)
    optima = np.stack([func._get_shift_vector(i + 1) for i in range(func.n_functions)])
    expected_values = np.asarray([func.f_global + bias for bias in func.biases])

    values = []
    for i, optimum in enumerate(optima):
        expected_weights = np.eye(func.n_functions)[i]

        np.testing.assert_allclose(func._compute_weights(optimum), expected_weights, atol=0.0)
        values.append(func(optimum))

    np.testing.assert_allclose(values, expected_values, atol=1e-8)
    np.testing.assert_allclose(func.batch(optima), expected_values, atol=1e-8)
