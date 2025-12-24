# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Multiplicative noise."""

from typing import Any, Dict, Optional

from ._base import BaseNoise


class MultiplicativeNoise(BaseNoise):
    """Multiplicative Gaussian noise: f(x) * (1 + N(0, sigma^2)).

    Applies noise proportional to the function value. Useful for
    simulating relative measurement uncertainty where larger values
    have proportionally larger noise.

    Parameters
    ----------
    sigma : float, default=0.1
        Standard deviation of the multiplicative factor. A value of 0.1
        means the noise factor is typically within +/-10% of 1.0.
        This is the initial value if a schedule is used.
    sigma_final : float, optional
        Final sigma when using a schedule. Defaults to sigma.
    seed : int, optional
        Random seed for reproducibility.
    schedule : str, optional
        Schedule for decaying noise. See BaseNoise for options.
    total_evaluations : int, optional
        Total evaluations for the schedule.

    Attributes
    ----------
    last_noise : float or None
        The multiplicative factor (not the final noise contribution)
        from the most recent apply() call. The actual noise added
        is value * last_noise.

    Examples
    --------
    Constant multiplicative noise (+/-10%):

    >>> noise = MultiplicativeNoise(sigma=0.1, seed=42)
    >>> noisy = noise.apply(100.0, {})
    >>> # Result is approximately 100 * (1 + small_gaussian)

    Note that for values near zero, multiplicative noise has minimal
    effect. Use GaussianNoise for additive noise that is independent
    of the function value.
    """

    def __init__(
        self,
        sigma: float = 0.1,
        sigma_final: Optional[float] = None,
        seed: Optional[int] = None,
        schedule: Optional[str] = None,
        total_evaluations: Optional[int] = None,
    ):
        super().__init__(
            seed=seed,
            schedule=schedule,
            total_evaluations=total_evaluations,
        )

        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")

        self._sigma_initial = sigma
        self._sigma_final = sigma_final if sigma_final is not None else sigma

        if self._sigma_final < 0:
            raise ValueError(f"sigma_final must be non-negative, got {sigma_final}")

    @property
    def sigma(self) -> float:
        """Current sigma based on schedule progress."""
        factor = self._get_schedule_factor()
        return self._sigma_final + (self._sigma_initial - self._sigma_final) * factor

    def _apply_noise(self, value: float, params: Dict[str, Any]) -> float:
        """Apply multiplicative noise to the value."""
        self.last_noise = self._rng.normal(0.0, self.sigma)
        return value * (1.0 + self.last_noise)
