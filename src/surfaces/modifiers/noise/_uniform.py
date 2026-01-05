# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Uniform (bounded additive) noise."""

from typing import Any, Dict, Optional

from ._base import BaseNoise


class UniformNoise(BaseNoise):
    """Additive uniform noise: f(x) + U(low, high).

    Adds uniformly distributed noise within a specified range.
    Supports optional scheduling to decay the noise range over evaluations.

    Parameters
    ----------
    low : float, default=-0.1
        Lower bound of the uniform distribution. Initial value if
        schedule is used.
    high : float, default=0.1
        Upper bound of the uniform distribution. Initial value if
        schedule is used.
    low_final : float, optional
        Final lower bound when using a schedule. Defaults to low.
    high_final : float, optional
        Final upper bound when using a schedule. Defaults to high.
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
    Constant noise in [-0.1, 0.1]:

    >>> noise = UniformNoise(low=-0.1, high=0.1, seed=42)
    >>> noisy = noise.apply(5.0, {})

    Symmetric noise (convenience):

    >>> noise = UniformNoise(low=-0.5, high=0.5)

    Decaying noise range:

    >>> noise = UniformNoise(
    ...     low=-0.5, high=0.5,
    ...     low_final=-0.01, high_final=0.01,
    ...     schedule="linear",
    ...     total_evaluations=1000
    ... )
    """

    def __init__(
        self,
        low: float = -0.1,
        high: float = 0.1,
        low_final: Optional[float] = None,
        high_final: Optional[float] = None,
        seed: Optional[int] = None,
        schedule: Optional[str] = None,
        total_evaluations: Optional[int] = None,
    ):
        super().__init__(
            seed=seed,
            schedule=schedule,
            total_evaluations=total_evaluations,
        )

        if low > high:
            raise ValueError(f"low ({low}) must be <= high ({high})")

        self._low_initial = low
        self._high_initial = high
        self._low_final = low_final if low_final is not None else low
        self._high_final = high_final if high_final is not None else high

        if self._low_final > self._high_final:
            raise ValueError(
                f"low_final ({self._low_final}) must be <= high_final ({self._high_final})"
            )

    @property
    def low(self) -> float:
        """Current lower bound based on schedule progress."""
        factor = self._get_schedule_factor()
        return self._low_final + (self._low_initial - self._low_final) * factor

    @property
    def high(self) -> float:
        """Current upper bound based on schedule progress."""
        factor = self._get_schedule_factor()
        return self._high_final + (self._high_initial - self._high_final) * factor

    def _apply_noise(self, value: float, params: Dict[str, Any]) -> float:
        """Apply uniform noise to the value."""
        self.last_noise = self._rng.uniform(self.low, self.high)
        return value + self.last_noise
