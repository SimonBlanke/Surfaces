# Tests for the noise module


import numpy as np
import pytest

from surfaces.modifiers import GaussianNoise, MultiplicativeNoise, UniformNoise
from surfaces.test_functions import SphereFunction


class TestGaussianNoise:
    """Tests for GaussianNoise."""

    def test_basic_noise_application(self):
        """Test that Gaussian noise is applied to values."""
        noise = GaussianNoise(sigma=0.1, seed=42)
        value = 5.0
        noisy = noise.apply(value, {}, {})

        assert noisy != value
        assert noise.last_noise is not None
        assert abs(noise.last_noise) < 1.0  # Reasonable for sigma=0.1

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same noise sequence."""
        noise1 = GaussianNoise(sigma=0.1, seed=42)
        noise2 = GaussianNoise(sigma=0.1, seed=42)

        values1 = [noise1.apply(1.0, {}, {}) for _ in range(10)]
        values2 = [noise2.apply(1.0, {}, {}) for _ in range(10)]

        assert values1 == values2

    def test_different_seeds_different_noise(self):
        """Test that different seeds produce different noise."""
        noise1 = GaussianNoise(sigma=0.1, seed=42)
        noise2 = GaussianNoise(sigma=0.1, seed=123)

        values1 = [noise1.apply(1.0, {}, {}) for _ in range(10)]
        values2 = [noise2.apply(1.0, {}, {}) for _ in range(10)]

        assert values1 != values2

    def test_sigma_affects_noise_magnitude(self):
        """Test that larger sigma produces larger noise."""
        noise_small = GaussianNoise(sigma=0.01, seed=42)
        noise_large = GaussianNoise(sigma=1.0, seed=42)

        # Apply many times and compute variance
        small_noises = [noise_small.apply(0.0, {}, {}) for _ in range(1000)]
        noise_small.reset()
        noise_large.reset()
        large_noises = [noise_large.apply(0.0, {}, {}) for _ in range(1000)]

        small_var = np.var(small_noises)
        large_var = np.var(large_noises)

        assert large_var > small_var * 10  # Much larger variance

    def test_negative_sigma_raises(self):
        """Test that negative sigma raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            GaussianNoise(sigma=-0.1)

    def test_reset_restores_initial_state(self):
        """Test that reset restores the initial random state."""
        noise = GaussianNoise(sigma=0.1, seed=42)

        values1 = [noise.apply(1.0, {}, {}) for _ in range(5)]
        noise.reset()
        values2 = [noise.apply(1.0, {}, {}) for _ in range(5)]

        assert values1 == values2
        assert noise.evaluation_count == 5


class TestUniformNoise:
    """Tests for UniformNoise."""

    def test_basic_noise_application(self):
        """Test that uniform noise is applied to values."""
        noise = UniformNoise(low=-0.1, high=0.1, seed=42)
        value = 5.0
        noisy = noise.apply(value, {}, {})

        assert noisy != value
        assert noise.last_noise is not None
        assert -0.1 <= noise.last_noise <= 0.1

    def test_noise_within_bounds(self):
        """Test that noise stays within specified bounds."""
        noise = UniformNoise(low=-0.5, high=0.5, seed=42)

        for _ in range(100):
            noise.apply(0.0, {}, {})
            assert -0.5 <= noise.last_noise <= 0.5

    def test_low_greater_than_high_raises(self):
        """Test that low > high raises ValueError."""
        with pytest.raises(ValueError, match="must be <="):
            UniformNoise(low=0.5, high=-0.5)

    def test_asymmetric_bounds(self):
        """Test noise with asymmetric bounds."""
        noise = UniformNoise(low=0.0, high=1.0, seed=42)

        noises = [noise.apply(0.0, {}, {}) for _ in range(100)]

        assert all(0.0 <= n <= 1.0 for n in noises)


class TestMultiplicativeNoise:
    """Tests for MultiplicativeNoise."""

    def test_basic_noise_application(self):
        """Test that multiplicative noise is applied."""
        noise = MultiplicativeNoise(sigma=0.1, seed=42)
        value = 100.0
        noisy = noise.apply(value, {}, {})

        assert noisy != value
        # Should be within roughly +/-30% for sigma=0.1
        assert 50.0 < noisy < 150.0

    def test_noise_scales_with_value(self):
        """Test that noise effect scales with function value."""
        noise = MultiplicativeNoise(sigma=0.1, seed=42)

        # Apply to small and large values
        small_noises = []
        large_noises = []

        for _ in range(100):
            noise.reset()
            small_result = noise.apply(1.0, {}, {})
            noise.reset()
            large_result = noise.apply(1000.0, {}, {})

            small_noises.append(abs(small_result - 1.0))
            large_noises.append(abs(large_result - 1000.0))

        # Large values should have larger absolute noise
        assert np.mean(large_noises) > np.mean(small_noises) * 100

    def test_zero_value_has_no_effect(self):
        """Test that multiplicative noise on zero returns zero."""
        noise = MultiplicativeNoise(sigma=0.5, seed=42)

        for _ in range(10):
            result = noise.apply(0.0, {}, {})
            assert result == 0.0


class TestNoiseSchedule:
    """Tests for noise schedule functionality."""

    def test_linear_schedule_decay(self):
        """Test that linear schedule decays sigma over evaluations."""
        noise = GaussianNoise(
            sigma=1.0,
            sigma_final=0.0,
            schedule="linear",
            total_evaluations=100,
            seed=42,
        )

        # At start, sigma should be 1.0
        assert noise.sigma == pytest.approx(1.0, rel=0.01)

        # Evaluate 50 times
        for _ in range(50):
            noise.apply(0.0, {}, {})

        # At halfway, sigma should be 0.5
        assert noise.sigma == pytest.approx(0.5, rel=0.01)

        # Evaluate 50 more times
        for _ in range(50):
            noise.apply(0.0, {}, {})

        # At end, sigma should be 0.0
        assert noise.sigma == pytest.approx(0.0, rel=0.01)

    def test_exponential_schedule_decay(self):
        """Test that exponential schedule decays faster initially."""
        noise = GaussianNoise(
            sigma=1.0,
            sigma_final=0.0,
            schedule="exponential",
            total_evaluations=100,
            seed=42,
        )

        # Evaluate halfway
        for _ in range(50):
            noise.apply(0.0, {}, {})

        # Exponential decay at halfway: exp(-5 * 0.5) = exp(-2.5) ≈ 0.082
        # sigma = 0.0 + (1.0 - 0.0) * 0.082 ≈ 0.082
        assert noise.sigma < 0.15

    def test_cosine_schedule_decay(self):
        """Test cosine annealing schedule."""
        noise = GaussianNoise(
            sigma=1.0,
            sigma_final=0.0,
            schedule="cosine",
            total_evaluations=100,
            seed=42,
        )

        # At start
        assert noise.sigma == pytest.approx(1.0, rel=0.01)

        # At halfway: 0.5 * (1 + cos(pi * 0.5)) = 0.5
        for _ in range(50):
            noise.apply(0.0, {}, {})
        assert noise.sigma == pytest.approx(0.5, rel=0.01)

        # At end: 0.5 * (1 + cos(pi)) = 0
        for _ in range(50):
            noise.apply(0.0, {}, {})
        assert noise.sigma == pytest.approx(0.0, rel=0.01)

    def test_schedule_requires_total_evaluations(self):
        """Test that schedule without total_evaluations raises."""
        with pytest.raises(ValueError, match="total_evaluations required"):
            GaussianNoise(sigma=1.0, schedule="linear")

    def test_invalid_schedule_raises(self):
        """Test that invalid schedule type raises."""
        with pytest.raises(ValueError, match="must be"):
            GaussianNoise(sigma=1.0, schedule="invalid", total_evaluations=100)

    def test_uniform_noise_schedule(self):
        """Test that UniformNoise also supports schedules."""
        noise = UniformNoise(
            low=-1.0,
            high=1.0,
            low_final=-0.1,
            high_final=0.1,
            schedule="linear",
            total_evaluations=100,
            seed=42,
        )

        # At start
        assert noise.low == pytest.approx(-1.0, rel=0.01)
        assert noise.high == pytest.approx(1.0, rel=0.01)

        # Evaluate to completion
        for _ in range(100):
            noise.apply(0.0, {}, {})

        # At end
        assert noise.low == pytest.approx(-0.1, rel=0.01)
        assert noise.high == pytest.approx(0.1, rel=0.01)


class TestNoiseIntegration:
    """Tests for noise integration with test functions."""

    def test_noise_applied_to_function(self):
        """Test that noise is applied when evaluating a function."""
        noise = GaussianNoise(sigma=0.1, seed=42)
        func = SphereFunction(n_dim=2, modifiers=[noise])

        result1 = func([0.0, 0.0])
        result2 = func([0.0, 0.0])

        # Results should differ due to noise (true value is 0.0)
        assert result1 != result2
        assert result1 != 0.0
        assert result2 != 0.0

    def test_true_value_bypasses_noise(self):
        """Test that true_value() returns value without noise."""
        noise = GaussianNoise(sigma=0.5, seed=42)
        func = SphereFunction(n_dim=2, modifiers=[noise])

        noisy = func([1.0, 2.0])
        true = func.true_value([1.0, 2.0])

        # True value should be 1^2 + 2^2 = 5.0
        assert true == pytest.approx(5.0, rel=1e-10)
        # Noisy value should differ
        assert noisy != true

    def test_last_noise_property(self):
        """Test that last_noise property works on test function."""
        noise = GaussianNoise(sigma=0.1, seed=42)
        func = SphereFunction(n_dim=2, modifiers=[noise])

        # Before any evaluation - noise is configured but not yet applied
        assert noise.last_noise is None

        # After evaluation
        func([0.0, 0.0])
        assert noise.last_noise is not None

    def test_modifiers_property(self):
        """Test that modifiers property returns the configured modifiers."""
        noise = GaussianNoise(sigma=0.1, seed=42)
        func = SphereFunction(n_dim=2, modifiers=[noise])

        assert noise in func.modifiers

    def test_no_noise_by_default(self):
        """Test that functions have no noise by default."""
        func = SphereFunction(n_dim=2)

        assert len(func.modifiers) == 0

        result1 = func([1.0, 2.0])
        result2 = func([1.0, 2.0])

        # Results should be identical without noise
        assert result1 == result2

    def test_reset_modifiers(self):
        """Test that reset_modifiers resets the modifiers state."""
        noise = GaussianNoise(sigma=0.1, seed=42)
        func = SphereFunction(n_dim=2, modifiers=[noise])

        # Evaluate a few times
        results1 = [func([0.0, 0.0]) for _ in range(5)]

        # Reset and evaluate again
        func.reset_modifiers()
        results2 = [func([0.0, 0.0]) for _ in range(5)]

        # Should get same sequence
        assert results1 == results2

    def test_noise_with_maximize_objective(self):
        """Test that noise works correctly with maximize objective."""
        noise = GaussianNoise(sigma=0.1, seed=42)
        func = SphereFunction(n_dim=2, objective="maximize", modifiers=[noise])

        # Evaluate
        result = func([1.0, 2.0])
        true = func.true_value([1.0, 2.0])

        # True value with maximize is -5.0
        assert true == pytest.approx(-5.0, rel=1e-10)
        # Noisy result should also be negative-ish
        assert result != true

    def test_noise_with_memory_caching(self):
        """Test interaction of noise with memory caching."""
        noise = GaussianNoise(sigma=0.1, seed=42)
        func = SphereFunction(n_dim=2, modifiers=[noise], memory=True)

        # First evaluation adds to cache
        result1 = func([1.0, 2.0])

        # Second evaluation should return cached (noisy) value
        result2 = func([1.0, 2.0])

        # Same cached value
        assert result1 == result2

        # Noise layer counter should have incremented only once
        assert noise.evaluation_count == 1


class TestEvaluationCounter:
    """Tests for the evaluation counter in noise layers."""

    def test_counter_increments(self):
        """Test that evaluation counter increments with each apply."""
        noise = GaussianNoise(sigma=0.1, seed=42)

        assert noise.evaluation_count == 0

        for i in range(5):
            noise.apply(1.0, {}, {})
            assert noise.evaluation_count == i + 1

    def test_counter_resets(self):
        """Test that counter resets with reset()."""
        noise = GaussianNoise(sigma=0.1, seed=42)

        for _ in range(10):
            noise.apply(1.0, {}, {})

        assert noise.evaluation_count == 10

        noise.reset()
        assert noise.evaluation_count == 0
