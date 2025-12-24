# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Base class for noise layers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BaseNoise(ABC):
    """Base class for noise layers that can be applied to test functions.

    Noise layers add stochastic disturbances to function evaluations,
    useful for testing algorithm robustness to noisy observations.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility. If None, uses non-deterministic
        random state.
    schedule : str, optional
        Schedule for decaying noise over evaluations. Options:
        - None: Constant noise (default)
        - "linear": Linear decay from initial to final
        - "exponential": Exponential decay
        - "cosine": Cosine annealing
    total_evaluations : int, optional
        Total number of evaluations for the schedule. Required if
        schedule is set.

    Attributes
    ----------
    last_noise : float or None
        The noise value from the most recent apply() call.
        None if apply() has not been called yet.

    Examples
    --------
    >>> from surfaces.noise import GaussianNoise
    >>> noise = GaussianNoise(sigma=0.1, seed=42)
    >>> noisy_value = noise.apply(5.0, {"x0": 0.5})
    >>> print(noise.last_noise)  # The noise that was added
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        schedule: Optional[str] = None,
        total_evaluations: Optional[int] = None,
    ):
        if schedule is not None and total_evaluations is None:
            raise ValueError("total_evaluations required when schedule is set")

        if schedule is not None and schedule not in ("linear", "exponential", "cosine"):
            raise ValueError(
                f"schedule must be 'linear', 'exponential', or 'cosine', got '{schedule}'"
            )

        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._schedule = schedule
        self._total_evaluations = total_evaluations
        self._evaluation_count = 0
        self.last_noise: Optional[float] = None

    def _get_schedule_factor(self) -> float:
        """Get the current schedule factor in [0, 1].

        Returns 1.0 at the start (full noise) and decays toward 0.0
        according to the schedule type.

        Returns
        -------
        float
            Schedule factor between 0 and 1.
        """
        if self._schedule is None:
            return 1.0

        progress = self._evaluation_count / self._total_evaluations
        progress = min(1.0, progress)  # Cap at 1.0

        if self._schedule == "linear":
            return 1.0 - progress
        elif self._schedule == "exponential":
            # Decays to ~0.0067 at progress=1.0
            return np.exp(-5.0 * progress)
        elif self._schedule == "cosine":
            # Smooth cosine decay
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        return 1.0

    def apply(self, value: float, params: Dict[str, Any]) -> float:
        """Apply noise to a function value.

        Parameters
        ----------
        value : float
            The original function value.
        params : dict
            The input parameters (available for heteroscedastic noise).

        Returns
        -------
        float
            The noisy function value.
        """
        self._evaluation_count += 1
        return self._apply_noise(value, params)

    @abstractmethod
    def _apply_noise(self, value: float, params: Dict[str, Any]) -> float:
        """Apply noise to a value. Override in subclasses.

        Parameters
        ----------
        value : float
            The original function value.
        params : dict
            The input parameters.

        Returns
        -------
        float
            The noisy function value.
        """
        pass

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the noise layer state.

        Resets the evaluation counter and random state.

        Parameters
        ----------
        seed : int, optional
            New seed for the random state. If None, uses the original seed.
        """
        self._evaluation_count = 0
        self._rng = np.random.default_rng(seed if seed is not None else self._seed)
        self.last_noise = None

    @property
    def evaluation_count(self) -> int:
        """Number of times apply() has been called."""
        return self._evaluation_count
