# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Gaussian (additive normal) noise."""

from typing import Any, Dict, Optional

from ._base import BaseNoise


class GaussianNoise(BaseNoise):
    """Additive Gaussian noise: f(x) + N(0, sigma^2).

    Adds normally distributed noise to function evaluations.
    Supports optional scheduling to decay sigma over evaluations.

    Parameters
    ----------
    sigma : float, default=0.1
        Standard deviation of the Gaussian noise. This is the initial
        value if a schedule is used.
    sigma_final : float, optional
        Final standard deviation when using a schedule. If None,
        defaults to sigma (no decay in the sigma value itself,
        but schedule factor still applies).
    seed : int, optional
        Random seed for reproducibility.
    schedule : str, optional
        Schedule for decaying noise. See BaseNoise for options.
    total_evaluations : int, optional
        Total evaluations for the schedule.

    Attributes
    ----------
    last_noise : float or None
        The noise value added in the most recent apply() call.

    Examples
    --------
    Constant noise:

    >>> noise = GaussianNoise(sigma=0.1, seed=42)
    >>> noisy = noise.apply(5.0, {})
    >>> print(f"Added noise: {noise.last_noise:.4f}")

    Decaying noise (sigma: 0.5 -> 0.01 over 1000 evaluations):

    >>> noise = GaussianNoise(
    ...     sigma=0.5,
    ...     sigma_final=0.01,
    ...     schedule="linear",
    ...     total_evaluations=1000,
    ...     seed=42
    ... )
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
        """Apply Gaussian noise to the value."""
        self.last_noise = self._rng.normal(0.0, self.sigma)
        return value + self.last_noise
