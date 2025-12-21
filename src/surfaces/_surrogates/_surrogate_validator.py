# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Surrogate model validation tools.

Compare surrogate predictions against real function evaluations
to assess approximation accuracy.
"""

import time
from itertools import product
from typing import Any, Dict, List, Optional

import numpy as np


class SurrogateValidator:
    """Validate surrogate model accuracy against real function.

    Parameters
    ----------
    function : MachineLearningFunction
        The function to validate. Must have use_surrogate=False.

    Examples
    --------
    >>> from surfaces import KNeighborsClassifierFunction
    >>> from surfaces._surrogates import SurrogateValidator
    >>>
    >>> func = KNeighborsClassifierFunction()
    >>> validator = SurrogateValidator(func)
    >>> results = validator.validate_random(n_samples=100)
    >>> print(results)
    """

    def __init__(self, function):
        self.function = function

        # Ensure we have both real and surrogate versions
        if getattr(function, "use_surrogate", False):
            raise ValueError(
                "Function should have use_surrogate=False. "
                "The validator will create its own surrogate instance."
            )

        # Create surrogate version
        func_class = type(function)
        self._surrogate_func = func_class(
            objective=function.objective,
            use_surrogate=True,
        )

        if not self._surrogate_func.use_surrogate:
            raise ValueError(f"No surrogate model available for {func_class.__name__}")

    def _get_search_space_values(self) -> Dict[str, List]:
        """Get search space as dict of lists."""
        return self.function.search_space

    def _sample_random(self, n_samples: int, seed: Optional[int] = None) -> List[Dict]:
        """Generate random parameter samples."""
        if seed is not None:
            np.random.seed(seed)

        search_space = self._get_search_space_values()
        samples = []

        for _ in range(n_samples):
            params = {}
            for name, values in search_space.items():
                params[name] = np.random.choice(values)
            samples.append(params)

        return samples

    def _sample_grid(self, max_samples: Optional[int] = None) -> List[Dict]:
        """Generate grid parameter samples."""
        search_space = self._get_search_space_values()
        param_names = sorted(search_space.keys())

        grid_points = list(product(*[search_space[name] for name in param_names]))

        if max_samples and len(grid_points) > max_samples:
            indices = np.random.choice(len(grid_points), max_samples, replace=False)
            grid_points = [grid_points[i] for i in indices]

        samples = []
        for point in grid_points:
            params = {name: val for name, val in zip(param_names, point)}
            samples.append(params)

        return samples

    def validate_random(
        self,
        n_samples: int = 100,
        seed: Optional[int] = 42,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Validate surrogate using random samples.

        Parameters
        ----------
        n_samples : int
            Number of random samples to evaluate.
        seed : int, optional
            Random seed for reproducibility.
        verbose : bool
            Print progress and results.

        Returns
        -------
        dict
            Validation results including metrics and raw data.
        """
        samples = self._sample_random(n_samples, seed)
        return self._validate(samples, "random", verbose)

    def validate_grid(
        self,
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Validate surrogate using grid samples.

        Parameters
        ----------
        max_samples : int, optional
            Maximum samples. If grid is larger, subsample randomly.
        verbose : bool
            Print progress and results.

        Returns
        -------
        dict
            Validation results including metrics and raw data.
        """
        samples = self._sample_grid(max_samples)
        return self._validate(samples, "grid", verbose)

    def _validate(
        self,
        samples: List[Dict],
        method: str,
        verbose: bool,
    ) -> Dict[str, Any]:
        """Run validation on given samples."""
        n_samples = len(samples)

        if verbose:
            print(f"Validating {n_samples} samples ({method} sampling)...")

        y_real = []
        y_surr = []
        errors = []
        real_times = []
        surr_times = []

        for i, params in enumerate(samples):
            # Real evaluation
            start = time.time()
            try:
                real = self.function.pure_objective_function(params)
                real_time = time.time() - start

                if np.isnan(real):
                    continue

                # Surrogate evaluation
                start = time.time()
                surr = self._surrogate_func._surrogate.predict(params)
                surr_time = time.time() - start

                y_real.append(real)
                y_surr.append(surr)
                errors.append(real - surr)
                real_times.append(real_time)
                surr_times.append(surr_time)

                if verbose and (i + 1) % 50 == 0:
                    print(f"  Evaluated {i + 1}/{n_samples} samples")

            except Exception as e:
                if verbose:
                    print(f"  Error at sample {i}: {e}")

        y_real = np.array(y_real)
        y_surr = np.array(y_surr)
        errors = np.array(errors)

        # Compute metrics
        n_valid = len(y_real)
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        max_error = np.max(np.abs(errors))

        # R² score
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Correlation
        correlation = np.corrcoef(y_real, y_surr)[0, 1]

        # Timing
        avg_real_time = np.mean(real_times) * 1000  # ms
        avg_surr_time = np.mean(surr_times) * 1000  # ms
        speedup = avg_real_time / avg_surr_time if avg_surr_time > 0 else 0

        results = {
            "method": method,
            "n_samples": n_valid,
            "n_skipped": n_samples - n_valid,
            "metrics": {
                "r2": r2,
                "mae": mae,
                "rmse": rmse,
                "max_error": max_error,
                "correlation": correlation,
            },
            "timing": {
                "avg_real_ms": avg_real_time,
                "avg_surrogate_ms": avg_surr_time,
                "speedup": speedup,
            },
            "data": {
                "y_real": y_real,
                "y_surrogate": y_surr,
                "errors": errors,
            },
        }

        if verbose:
            self._print_results(results)

        return results

    def _print_results(self, results: Dict[str, Any]) -> None:
        """Print formatted validation results."""
        m = results["metrics"]
        t = results["timing"]

        print(f"\n{'=' * 50}")
        print(f"Surrogate Validation Results ({results['method']} sampling)")
        print(f"{'=' * 50}")
        print(f"Samples evaluated: {results['n_samples']}")
        if results["n_skipped"] > 0:
            print(f"Samples skipped:   {results['n_skipped']}")

        print(f"\n{'Accuracy Metrics':}")
        print(f"  R² Score:     {m['r2']:.4f}")
        print(f"  Correlation:  {m['correlation']:.4f}")
        print(f"  MAE:          {m['mae']:.4f}")
        print(f"  RMSE:         {m['rmse']:.4f}")
        print(f"  Max Error:    {m['max_error']:.4f}")

        print(f"\n{'Timing':}")
        print(f"  Real function:  {t['avg_real_ms']:>8.2f} ms/eval")
        print(f"  Surrogate:      {t['avg_surrogate_ms']:>8.2f} ms/eval")
        print(f"  Speedup:        {t['speedup']:>8.0f}x")

    def summary(
        self,
        n_random: int = 100,
        n_grid: int = 100,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Run both random and grid validation.

        Parameters
        ----------
        n_random : int
            Number of random samples.
        n_grid : int
            Maximum grid samples.
        seed : int
            Random seed.

        Returns
        -------
        dict
            Combined results from both methods.
        """
        print("Running validation summary...\n")

        random_results = self.validate_random(n_random, seed, verbose=True)
        print()
        grid_results = self.validate_grid(n_grid, verbose=True)

        return {
            "random": random_results,
            "grid": grid_results,
        }
