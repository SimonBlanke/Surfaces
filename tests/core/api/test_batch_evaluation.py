# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for batch evaluation functionality."""

import numpy as np
import pytest

from surfaces.test_functions.algebraic import (
    AckleyFunction,
    BealeFunction,
    BoothFunction,
    BukinFunctionN6,
    CrossInTrayFunction,
    DampedSineFunction,
    DropWaveFunction,
    EasomFunction,
    EggholderFunction,
    ForresterFunction,
    GoldsteinPriceFunction,
    GramacyAndLeeFunction,
    GriewankFunction,
    HimmelblausFunction,
    HölderTableFunction,
    LangermannFunction,
    LeviFunctionN13,
    MatyasFunction,
    McCormickFunction,
    QuadraticExponentialFunction,
    RastriginFunction,
    RosenbrockFunction,
    SchafferFunctionN2,
    SimionescuFunction,
    SineProductFunction,
    SphereFunction,
    StyblinskiTangFunction,
    ThreeHumpCamelFunction,
    # Function lists for comprehensive tests
    constrained_functions,
    standard_functions_1d,
    standard_functions_2d,
    standard_functions_nd,
)
from surfaces.test_functions.benchmark.bbob import (
    # Separable (f1-f5)
    BuecheRastrigin,
    EllipsoidalSeparable,
    LinearSlope,
    RastriginSeparable,
    Sphere as BBOBSphere,
    # Low/Moderate Conditioning (f6-f9)
    AttractiveSector,
    StepEllipsoidal,
    RosenbrockOriginal,
    RosenbrockRotated,
    # High Conditioning (f10-f14)
    EllipsoidalRotated,
    Discus,
    BentCigar,
    SharpRidge,
    DifferentPowers,
    # Multimodal Adequate (f15-f19)
    RastriginRotated,
    Weierstrass,
    SchaffersF7,
    SchaffersF7Ill,
    GriewankRosenbrock,
    # Multimodal Weak (f20-f24)
    Schwefel,
    Gallagher101,
    Gallagher21,
    Katsuura,
    LunacekBiRastrigin,
)


class TestBatchEvaluationBasic:
    """Basic batch evaluation tests."""

    def test_sphere_batch_shape(self):
        """Batch returns correct shape."""
        func = SphereFunction(n_dim=3)
        X = np.random.randn(100, 3)
        results = func.batch(X)

        assert results.shape == (100,)

    def test_rastrigin_batch_shape(self):
        """Batch returns correct shape."""
        func = RastriginFunction(n_dim=5)
        X = np.random.randn(50, 5)
        results = func.batch(X)

        assert results.shape == (50,)

    def test_ackley_batch_shape(self):
        """Batch returns correct shape for 2D Ackley."""
        func = AckleyFunction()
        X = np.random.randn(200, 2)
        results = func.batch(X)

        assert results.shape == (200,)

    def test_rosenbrock_batch_shape(self):
        """Batch returns correct shape."""
        func = RosenbrockFunction(n_dim=4)
        X = np.random.randn(100, 4)
        results = func.batch(X)

        assert results.shape == (100,)

    def test_griewank_batch_shape(self):
        """Batch returns correct shape."""
        func = GriewankFunction(n_dim=6)
        X = np.random.randn(80, 6)
        results = func.batch(X)

        assert results.shape == (80,)

    def test_styblinski_tang_batch_shape(self):
        """Batch returns correct shape."""
        func = StyblinskiTangFunction(n_dim=3)
        X = np.random.randn(120, 3)
        results = func.batch(X)

        assert results.shape == (120,)


class TestBatchEvaluationCorrectness:
    """Tests that batch produces same results as sequential evaluation."""

    def test_sphere_batch_matches_sequential(self):
        """Batch results match sequential __call__ results."""
        func = SphereFunction(n_dim=3)
        X = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 3.0, 4.0],
            [-1.0, -2.0, -3.0],
        ])

        batch_results = func.batch(X)

        sequential_results = []
        for row in X:
            result = func(row)
            sequential_results.append(result)

        np.testing.assert_allclose(batch_results, sequential_results, rtol=1e-10)

    def test_rastrigin_batch_matches_sequential(self):
        """Batch results match sequential __call__ results."""
        func = RastriginFunction(n_dim=2)
        X = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, -0.5],
            [-2.0, 2.0],
        ])

        batch_results = func.batch(X)

        sequential_results = []
        for row in X:
            result = func(row)
            sequential_results.append(result)

        np.testing.assert_allclose(batch_results, sequential_results, rtol=1e-10)

    def test_ackley_batch_matches_sequential(self):
        """Batch results match sequential __call__ results."""
        func = AckleyFunction()
        X = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [-1.0, 2.0],
            [3.0, -3.0],
        ])

        batch_results = func.batch(X)

        sequential_results = []
        for row in X:
            result = func(row)
            sequential_results.append(result)

        np.testing.assert_allclose(batch_results, sequential_results, rtol=1e-10)

    def test_rosenbrock_batch_matches_sequential(self):
        """Batch results match sequential __call__ results."""
        func = RosenbrockFunction(n_dim=3)
        X = np.array([
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [2.0, 4.0, 8.0],
            [-1.0, 1.0, -1.0],
        ])

        batch_results = func.batch(X)

        sequential_results = []
        for row in X:
            result = func(row)
            sequential_results.append(result)

        np.testing.assert_allclose(batch_results, sequential_results, rtol=1e-10)

    def test_griewank_batch_matches_sequential(self):
        """Batch results match sequential __call__ results."""
        func = GriewankFunction(n_dim=3)
        X = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [10.0, -10.0, 5.0],
            [-5.0, 5.0, -5.0],
        ])

        batch_results = func.batch(X)

        sequential_results = []
        for row in X:
            result = func(row)
            sequential_results.append(result)

        np.testing.assert_allclose(batch_results, sequential_results, rtol=1e-10)

    def test_styblinski_tang_batch_matches_sequential(self):
        """Batch results match sequential __call__ results."""
        func = StyblinskiTangFunction(n_dim=2)
        X = np.array([
            [-2.903534, -2.903534],
            [0.0, 0.0],
            [1.0, -1.0],
            [-3.0, 3.0],
        ])

        batch_results = func.batch(X)

        sequential_results = []
        for row in X:
            result = func(row)
            sequential_results.append(result)

        np.testing.assert_allclose(batch_results, sequential_results, rtol=1e-10)


class TestBatchEvaluationGlobalOptimum:
    """Tests that batch correctly evaluates at global optimum."""

    def test_sphere_global_optimum(self):
        """Sphere at origin should be 0."""
        func = SphereFunction(n_dim=5)
        X = np.zeros((1, 5))
        results = func.batch(X)

        np.testing.assert_allclose(results[0], 0.0, atol=1e-10)

    def test_rastrigin_global_optimum(self):
        """Rastrigin at origin should be 0."""
        func = RastriginFunction(n_dim=3)
        X = np.zeros((1, 3))
        results = func.batch(X)

        np.testing.assert_allclose(results[0], 0.0, atol=1e-10)

    def test_ackley_global_optimum(self):
        """Ackley at origin should be 0."""
        func = AckleyFunction()
        X = np.zeros((1, 2))
        results = func.batch(X)

        np.testing.assert_allclose(results[0], 0.0, atol=1e-10)

    def test_rosenbrock_global_optimum(self):
        """Rosenbrock at (1, 1, ..., 1) should be 0."""
        func = RosenbrockFunction(n_dim=4)
        X = np.ones((1, 4))
        results = func.batch(X)

        np.testing.assert_allclose(results[0], 0.0, atol=1e-10)

    def test_griewank_global_optimum(self):
        """Griewank at origin should be 0."""
        func = GriewankFunction(n_dim=3)
        X = np.zeros((1, 3))
        results = func.batch(X)

        np.testing.assert_allclose(results[0], 0.0, atol=1e-10)

    def test_styblinski_tang_global_optimum(self):
        """Styblinski-Tang at (-2.903534, ...) should be approx -39.16617*n."""
        n_dim = 3
        func = StyblinskiTangFunction(n_dim=n_dim)
        X = np.full((1, n_dim), -2.903534)
        results = func.batch(X)

        expected = -39.16617 * n_dim
        np.testing.assert_allclose(results[0], expected, rtol=1e-5)


class TestBatchEvaluationParameters:
    """Tests for function parameters affecting batch evaluation."""

    def test_sphere_with_custom_A(self):
        """Sphere with custom A parameter."""
        func = SphereFunction(n_dim=2, A=2.0)
        X = np.array([[1.0, 1.0]])
        result = func.batch(X)

        # A * (1^2 + 1^2) = 2 * 2 = 4
        np.testing.assert_allclose(result[0], 4.0, rtol=1e-10)

    def test_rastrigin_with_custom_A(self):
        """Rastrigin with custom A parameter."""
        func = RastriginFunction(n_dim=2, A=5.0)
        X = np.zeros((1, 2))
        result = func.batch(X)

        # At origin: A*n + 0 - A*cos(0) - A*cos(0) = 5*2 + 0 - 5*1 - 5*1 = 0
        np.testing.assert_allclose(result[0], 0.0, atol=1e-10)

    def test_objective_maximize(self):
        """Batch respects maximize objective."""
        func_min = SphereFunction(n_dim=2, objective="minimize")
        func_max = SphereFunction(n_dim=2, objective="maximize")

        X = np.array([[1.0, 2.0]])

        result_min = func_min.batch(X)
        result_max = func_max.batch(X)

        np.testing.assert_allclose(result_max, -result_min, rtol=1e-10)


class TestBatchEvaluationErrors:
    """Tests for error handling in batch evaluation."""

    def test_wrong_dimensions(self):
        """Error when X has wrong number of dimensions."""
        func = SphereFunction(n_dim=3)
        X = np.random.randn(10, 5)  # Wrong: 5 dims instead of 3

        with pytest.raises(ValueError, match="Expected 3 dimensions"):
            func.batch(X)

    def test_1d_array_error(self):
        """Error when X is 1D instead of 2D."""
        func = SphereFunction(n_dim=3)
        X = np.array([1.0, 2.0, 3.0])  # 1D array

        with pytest.raises(ValueError, match="Expected 2D array"):
            func.batch(X)

    def test_not_array_like_error(self):
        """Error when input is not array-like."""
        func = SphereFunction(n_dim=2)

        with pytest.raises(TypeError, match="Expected array-like"):
            func.batch([{"x0": 1.0, "x1": 2.0}])


class TestBatchEvaluationPerformance:
    """Tests verifying batch is faster than sequential for large inputs."""

    def test_batch_produces_results_for_large_input(self):
        """Batch handles large inputs correctly."""
        func = SphereFunction(n_dim=10)
        X = np.random.randn(10000, 10)

        results = func.batch(X)

        assert results.shape == (10000,)
        assert not np.any(np.isnan(results))
        assert not np.any(np.isinf(results))


class TestBatchEvaluationArrayTypes:
    """Tests for different array input types."""

    def test_float32_input(self):
        """Batch works with float32 arrays."""
        func = SphereFunction(n_dim=2)
        X = np.array([[1.0, 2.0]], dtype=np.float32)
        results = func.batch(X)

        assert results.dtype == np.float32

    def test_float64_input(self):
        """Batch works with float64 arrays."""
        func = SphereFunction(n_dim=2)
        X = np.array([[1.0, 2.0]], dtype=np.float64)
        results = func.batch(X)

        assert results.dtype == np.float64


class TestBatch2DFunctions:
    """Tests for all 2D function batch implementations."""

    @pytest.mark.parametrize(
        "func_class,test_points",
        [
            (BoothFunction, [[1.0, 3.0], [0.0, 0.0], [2.0, -1.0]]),
            (MatyasFunction, [[0.0, 0.0], [1.0, 1.0], [-2.0, 2.0]]),
            (ThreeHumpCamelFunction, [[0.0, 0.0], [1.0, 1.0], [-1.0, 0.5]]),
            (BealeFunction, [[3.0, 0.5], [0.0, 0.0], [1.0, 1.0]]),
            (HimmelblausFunction, [[3.0, 2.0], [0.0, 0.0], [-2.0, 3.0]]),
            (GoldsteinPriceFunction, [[0.0, -1.0], [0.0, 0.0], [1.0, 1.0]]),
            (EasomFunction, [[np.pi, np.pi], [0.0, 0.0], [3.0, 3.0]]),
            (DropWaveFunction, [[0.0, 0.0], [1.0, 1.0], [-2.0, 2.0]]),
            (CrossInTrayFunction, [[1.34941, 1.34941], [0.0, 0.0], [5.0, 5.0]]),
            (HölderTableFunction, [[8.05502, 9.66459], [0.0, 0.0], [5.0, 5.0]]),
            (EggholderFunction, [[512.0, 404.2319], [0.0, 0.0], [100.0, 100.0]]),
            (SchafferFunctionN2, [[0.0, 0.0], [1.0, 1.0], [10.0, -10.0]]),
            (LeviFunctionN13, [[1.0, 1.0], [0.0, 0.0], [5.0, 5.0]]),
            (McCormickFunction, [[-0.54719, -1.54719], [0.0, 0.0], [2.0, 2.0]]),
            (BukinFunctionN6, [[-10.0, 1.0], [-5.0, 0.0], [0.0, 0.0]]),
            (LangermannFunction, [[0.0, 0.0], [3.0, 5.0], [5.0, 2.0]]),
        ],
    )
    def test_2d_batch_matches_sequential(self, func_class, test_points):
        """Batch results match sequential __call__ results for 2D functions."""
        func = func_class()
        X = np.array(test_points)

        batch_results = func.batch(X)

        sequential_results = []
        for row in X:
            result = func(row)
            sequential_results.append(result)

        np.testing.assert_allclose(batch_results, sequential_results, rtol=1e-9)

    def test_simionescu_batch_matches_sequential(self):
        """Simionescu with constraint handling."""
        func = SimionescuFunction()
        # Points inside constraint
        X = np.array([
            [0.5, 0.5],
            [-0.5, 0.5],
            [0.0, 0.0],
        ])

        batch_results = func.batch(X)

        sequential_results = []
        for row in X:
            result = func(row)
            sequential_results.append(result)

        np.testing.assert_allclose(batch_results, sequential_results, rtol=1e-9)

    def test_simionescu_constraint_nan(self):
        """Simionescu returns NaN for out-of-bounds points."""
        func = SimionescuFunction()
        # Point clearly outside constraint (r > 1.2)
        X = np.array([[2.0, 2.0]])

        results = func.batch(X)

        assert np.isnan(results[0])

    @pytest.mark.parametrize(
        "func_class,global_opt,expected",
        [
            (BoothFunction, [1.0, 3.0], 0.0),
            (MatyasFunction, [0.0, 0.0], 0.0),
            (ThreeHumpCamelFunction, [0.0, 0.0], 0.0),
            (BealeFunction, [3.0, 0.5], 0.0),
            (HimmelblausFunction, [3.0, 2.0], 0.0),
            (GoldsteinPriceFunction, [0.0, -1.0], 3.0),
            (DropWaveFunction, [0.0, 0.0], -1.0),
            (SchafferFunctionN2, [0.0, 0.0], 0.0),
        ],
    )
    def test_2d_global_optima(self, func_class, global_opt, expected):
        """Test global optima for 2D functions."""
        func = func_class()
        X = np.array([global_opt])

        results = func.batch(X)

        np.testing.assert_allclose(results[0], expected, atol=1e-9)


class TestBatch1DFunctions:
    """Tests for all 1D function batch implementations."""

    @pytest.mark.parametrize(
        "func_class,test_points",
        [
            (ForresterFunction, [[0.0], [0.5], [0.757], [1.0]]),
            (GramacyAndLeeFunction, [[0.5], [1.0], [1.5], [2.5]]),
            (DampedSineFunction, [[0.0], [0.68], [-1.0], [2.0]]),
            (SineProductFunction, [[0.1], [5.0], [7.98], [10.0]]),
            (QuadraticExponentialFunction, [[1.9], [2.87], [3.0], [3.9]]),
        ],
    )
    def test_1d_batch_matches_sequential(self, func_class, test_points):
        """Batch results match sequential __call__ results for 1D functions."""
        func = func_class()
        X = np.array(test_points)

        batch_results = func.batch(X)

        sequential_results = []
        for row in X:
            result = func(row)
            sequential_results.append(result)

        np.testing.assert_allclose(batch_results, sequential_results, rtol=1e-9)

    def test_1d_batch_shape(self):
        """Batch returns correct shape for 1D functions."""
        func = ForresterFunction()
        X = np.random.rand(100, 1)
        results = func.batch(X)

        assert results.shape == (100,)

    @pytest.mark.parametrize(
        "func_class,global_opt,expected",
        [
            (ForresterFunction, [0.7572487144081974], -6.020740055766075),
            (GramacyAndLeeFunction, [0.548563444114526], -0.869011134989500),
            (DampedSineFunction, [0.6795787635255166], -0.8242393984760573),
            (SineProductFunction, [7.9786653537049483], -7.916727371587256),
            (QuadraticExponentialFunction, [2.8680325095605212], -3.8504507087979953),
        ],
    )
    def test_1d_global_optima(self, func_class, global_opt, expected):
        """Test global optima for 1D functions."""
        func = func_class()
        X = np.array([global_opt])

        results = func.batch(X)

        np.testing.assert_allclose(results[0], expected, rtol=1e-9)

    def test_1d_large_batch(self):
        """1D functions handle large batch sizes."""
        func = ForresterFunction()
        X = np.linspace(0, 1, 10000).reshape(-1, 1)

        results = func.batch(X)

        assert results.shape == (10000,)
        assert not np.any(np.isnan(results))
        assert not np.any(np.isinf(results))


class TestBatchBBOBSeparable:
    """Tests for BBOB separable function batch implementations (f1-f5)."""

    @pytest.mark.parametrize("n_dim", [2, 5, 10])
    def test_bbob_sphere_batch_shape(self, n_dim):
        """BBOB Sphere batch returns correct shape."""
        func = BBOBSphere(n_dim=n_dim)
        X = np.random.randn(100, n_dim)
        results = func.batch(X)

        assert results.shape == (100,)

    @pytest.mark.parametrize(
        "func_class",
        [BBOBSphere, EllipsoidalSeparable, RastriginSeparable, BuecheRastrigin, LinearSlope],
    )
    def test_bbob_separable_batch_matches_sequential(self, func_class):
        """Batch results match sequential __call__ results for BBOB separable."""
        n_dim = 5
        func = func_class(n_dim=n_dim, instance=1)

        # Use random points within bounds
        rng = np.random.RandomState(42)
        X = rng.uniform(-4, 4, size=(20, n_dim))

        batch_results = func.batch(X)

        sequential_results = []
        for row in X:
            result = func(row)
            sequential_results.append(result)

        np.testing.assert_allclose(batch_results, sequential_results, rtol=1e-10)

    @pytest.mark.parametrize(
        "func_class",
        [BBOBSphere, EllipsoidalSeparable, RastriginSeparable, BuecheRastrigin, LinearSlope],
    )
    def test_bbob_separable_at_optimum(self, func_class):
        """Batch evaluation at optimum matches f_global."""
        n_dim = 5
        func = func_class(n_dim=n_dim, instance=1)

        # Evaluate at the optimum
        X = np.array([func.x_opt])
        results = func.batch(X)

        np.testing.assert_allclose(results[0], func.f_global, rtol=1e-9)

    def test_bbob_sphere_different_instances(self):
        """Different instances produce different results."""
        n_dim = 5
        func1 = BBOBSphere(n_dim=n_dim, instance=1)
        func2 = BBOBSphere(n_dim=n_dim, instance=2)

        X = np.random.RandomState(42).randn(10, n_dim)

        results1 = func1.batch(X)
        results2 = func2.batch(X)

        # Different instances should give different results
        # (due to different x_opt and f_opt)
        assert not np.allclose(results1, results2)

    @pytest.mark.parametrize(
        "func_class",
        [BBOBSphere, EllipsoidalSeparable, RastriginSeparable, BuecheRastrigin, LinearSlope],
    )
    def test_bbob_separable_large_batch(self, func_class):
        """BBOB separable handles large batch sizes."""
        n_dim = 10
        func = func_class(n_dim=n_dim, instance=1)

        X = np.random.randn(5000, n_dim)
        results = func.batch(X)

        assert results.shape == (5000,)
        assert not np.any(np.isnan(results))
        assert not np.any(np.isinf(results))

    def test_bbob_objective_maximize(self):
        """BBOB batch respects maximize objective."""
        n_dim = 5
        func_min = BBOBSphere(n_dim=n_dim, instance=1, objective="minimize")
        func_max = BBOBSphere(n_dim=n_dim, instance=1, objective="maximize")

        X = np.random.RandomState(42).randn(10, n_dim)

        result_min = func_min.batch(X)
        result_max = func_max.batch(X)

        np.testing.assert_allclose(result_max, -result_min, rtol=1e-10)


class TestBatchBBOBLowConditioning:
    """Tests for BBOB low/moderate conditioning functions (f6-f9)."""

    @pytest.mark.parametrize(
        "func_class",
        [AttractiveSector, StepEllipsoidal, RosenbrockOriginal, RosenbrockRotated],
    )
    def test_bbob_low_cond_batch_matches_sequential(self, func_class):
        """Batch results match sequential __call__ results."""
        n_dim = 5
        func = func_class(n_dim=n_dim, instance=1)

        rng = np.random.RandomState(42)
        X = rng.uniform(-4, 4, size=(20, n_dim))

        batch_results = func.batch(X)

        sequential_results = []
        for row in X:
            result = func(row)
            sequential_results.append(result)

        np.testing.assert_allclose(batch_results, sequential_results, rtol=1e-9)

    @pytest.mark.parametrize(
        "func_class",
        # RosenbrockRotated excluded: x_opt is not at global optimum due to rotated structure
        [AttractiveSector, StepEllipsoidal, RosenbrockOriginal],
    )
    def test_bbob_low_cond_at_optimum(self, func_class):
        """Batch evaluation at optimum matches f_global."""
        n_dim = 5
        func = func_class(n_dim=n_dim, instance=1)

        X = np.array([func.x_opt])
        results = func.batch(X)

        np.testing.assert_allclose(results[0], func.f_global, rtol=1e-8)


class TestBatchBBOBHighConditioning:
    """Tests for BBOB high conditioning functions (f10-f14)."""

    @pytest.mark.parametrize(
        "func_class",
        [EllipsoidalRotated, Discus, BentCigar, SharpRidge, DifferentPowers],
    )
    def test_bbob_high_cond_batch_matches_sequential(self, func_class):
        """Batch results match sequential __call__ results."""
        n_dim = 5
        func = func_class(n_dim=n_dim, instance=1)

        rng = np.random.RandomState(42)
        X = rng.uniform(-4, 4, size=(20, n_dim))

        batch_results = func.batch(X)

        sequential_results = []
        for row in X:
            result = func(row)
            sequential_results.append(result)

        np.testing.assert_allclose(batch_results, sequential_results, rtol=1e-9)

    @pytest.mark.parametrize(
        "func_class",
        [EllipsoidalRotated, Discus, BentCigar, SharpRidge, DifferentPowers],
    )
    def test_bbob_high_cond_at_optimum(self, func_class):
        """Batch evaluation at optimum matches f_global."""
        n_dim = 5
        func = func_class(n_dim=n_dim, instance=1)

        X = np.array([func.x_opt])
        results = func.batch(X)

        np.testing.assert_allclose(results[0], func.f_global, rtol=1e-8)


class TestBatchBBOBMultimodalAdequate:
    """Tests for BBOB multimodal adequate structure functions (f15-f19)."""

    @pytest.mark.parametrize(
        "func_class",
        [RastriginRotated, Weierstrass, SchaffersF7, SchaffersF7Ill, GriewankRosenbrock],
    )
    def test_bbob_multimodal_adequate_batch_matches_sequential(self, func_class):
        """Batch results match sequential __call__ results."""
        n_dim = 5
        func = func_class(n_dim=n_dim, instance=1)

        rng = np.random.RandomState(42)
        X = rng.uniform(-4, 4, size=(20, n_dim))

        batch_results = func.batch(X)

        sequential_results = []
        for row in X:
            result = func(row)
            sequential_results.append(result)

        np.testing.assert_allclose(batch_results, sequential_results, rtol=1e-8)

    @pytest.mark.parametrize(
        "func_class",
        [RastriginRotated, Weierstrass, SchaffersF7, SchaffersF7Ill, GriewankRosenbrock],
    )
    def test_bbob_multimodal_adequate_at_optimum(self, func_class):
        """Batch evaluation at optimum matches f_global."""
        n_dim = 5
        func = func_class(n_dim=n_dim, instance=1)

        X = np.array([func.x_opt])
        results = func.batch(X)

        np.testing.assert_allclose(results[0], func.f_global, rtol=1e-6)


class TestBatchBBOBMultimodalWeak:
    """Tests for BBOB multimodal weak structure functions (f20-f24)."""

    @pytest.mark.parametrize(
        "func_class",
        [Schwefel, Gallagher101, Gallagher21, Katsuura, LunacekBiRastrigin],
    )
    def test_bbob_multimodal_weak_batch_matches_sequential(self, func_class):
        """Batch results match sequential __call__ results."""
        n_dim = 5
        func = func_class(n_dim=n_dim, instance=1)

        rng = np.random.RandomState(42)
        X = rng.uniform(-4, 4, size=(20, n_dim))

        batch_results = func.batch(X)

        sequential_results = []
        for row in X:
            result = func(row)
            sequential_results.append(result)

        np.testing.assert_allclose(batch_results, sequential_results, rtol=1e-8)

    @pytest.mark.parametrize(
        "func_class",
        [Schwefel, Gallagher101, Gallagher21, Katsuura, LunacekBiRastrigin],
    )
    def test_bbob_multimodal_weak_at_optimum(self, func_class):
        """Batch evaluation at optimum matches f_global."""
        n_dim = 5
        func = func_class(n_dim=n_dim, instance=1)

        X = np.array([func.x_opt])
        results = func.batch(X)

        np.testing.assert_allclose(results[0], func.f_global, rtol=1e-6)


class TestBatchBBOBAllFunctions:
    """Comprehensive tests covering all 24 BBOB functions."""

    BBOB_SEPARABLE = [BBOBSphere, EllipsoidalSeparable, RastriginSeparable, BuecheRastrigin, LinearSlope]
    BBOB_LOW_COND = [AttractiveSector, StepEllipsoidal, RosenbrockOriginal, RosenbrockRotated]
    BBOB_HIGH_COND = [EllipsoidalRotated, Discus, BentCigar, SharpRidge, DifferentPowers]
    BBOB_MULTIMODAL_ADQ = [RastriginRotated, Weierstrass, SchaffersF7, SchaffersF7Ill, GriewankRosenbrock]
    BBOB_MULTIMODAL_WEAK = [Schwefel, Gallagher101, Gallagher21, Katsuura, LunacekBiRastrigin]

    ALL_BBOB = BBOB_SEPARABLE + BBOB_LOW_COND + BBOB_HIGH_COND + BBOB_MULTIMODAL_ADQ + BBOB_MULTIMODAL_WEAK

    @pytest.mark.parametrize("func_class", ALL_BBOB)
    @pytest.mark.parametrize("n_dim", [2, 5, 10])
    def test_bbob_batch_shape(self, func_class, n_dim):
        """All BBOB functions return correct batch shape."""
        func = func_class(n_dim=n_dim, instance=1)
        X = np.random.randn(50, n_dim)
        results = func.batch(X)

        assert results.shape == (50,)

    @pytest.mark.parametrize("func_class", ALL_BBOB)
    def test_bbob_no_nan_inf(self, func_class):
        """All BBOB batch results are finite."""
        n_dim = 5
        func = func_class(n_dim=n_dim, instance=1)

        rng = np.random.RandomState(42)
        X = rng.uniform(-4, 4, size=(100, n_dim))
        results = func.batch(X)

        assert not np.any(np.isnan(results)), f"{func_class.__name__} produced NaN"
        assert not np.any(np.isinf(results)), f"{func_class.__name__} produced Inf"

    @pytest.mark.parametrize("func_class", ALL_BBOB)
    @pytest.mark.parametrize("instance", [1, 2, 3])
    def test_bbob_different_instances(self, func_class, instance):
        """Different instances are correctly handled."""
        n_dim = 5
        func = func_class(n_dim=n_dim, instance=instance)

        X = np.array([func.x_opt])
        results = func.batch(X)

        np.testing.assert_allclose(
            results[0], func.f_global, rtol=1e-6,
            err_msg=f"{func_class.__name__} instance {instance} failed at optimum"
        )


class TestBatchEvaluationComprehensive:
    """Comprehensive smoke tests for ALL algebraic test functions.

    These tests ensure that every test function correctly supports batch evaluation
    by verifying output shape and finiteness.
    """

    @pytest.mark.parametrize("func_class", standard_functions_1d)
    def test_1d_batch_shape(self, func_class):
        """All 1D functions return correct batch shape."""
        func = func_class()
        X = np.random.uniform(-1, 1, (50, 1))
        results = func.batch(X)

        assert results.shape == (50,), f"{func_class.__name__} returned wrong shape"
        assert np.all(np.isfinite(results)), f"{func_class.__name__} produced non-finite values"

    @pytest.mark.parametrize("func_class", standard_functions_2d)
    def test_2d_batch_shape(self, func_class):
        """All 2D functions return correct batch shape."""
        func = func_class()
        X = np.random.uniform(-1, 1, (50, 2))
        results = func.batch(X)

        assert results.shape == (50,), f"{func_class.__name__} returned wrong shape"
        # Note: Some 2D functions (SimionescuFunction) return NaN for out-of-bounds points

    @pytest.mark.parametrize("func_class", standard_functions_nd)
    @pytest.mark.parametrize("n_dim", [2, 5, 10])
    def test_nd_batch_shape(self, func_class, n_dim):
        """All N-D functions return correct batch shape across dimensions."""
        func = func_class(n_dim=n_dim)
        X = np.random.uniform(-1, 1, (50, n_dim))
        results = func.batch(X)

        assert results.shape == (50,), f"{func_class.__name__} returned wrong shape for n_dim={n_dim}"
        assert np.all(np.isfinite(results)), f"{func_class.__name__} produced non-finite values"

    @pytest.mark.parametrize("func_class", constrained_functions)
    def test_constrained_batch_shape(self, func_class):
        """All constrained (engineering) functions return correct batch shape."""
        func = func_class()
        # Use small positive values - engineering functions often have positive bounds
        X = np.random.uniform(0.1, 1.0, (50, func.n_dim))
        results = func.batch(X)

        assert results.shape == (50,), f"{func_class.__name__} returned wrong shape"
