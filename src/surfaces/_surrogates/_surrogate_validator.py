# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Surrogate model validation tools.

This module provides tools for comparing surrogate model predictions against
real function evaluations to assess approximation accuracy and speedup.

Architecture Overview
---------------------

The validator works with ML functions that have two evaluation modes:

1. **Real evaluation** (use_surrogate=False):
   - Calls `pure_objective_function(params)` which performs actual ML training
   - `params` contains only hyperparameters (e.g., n_neighbors, algorithm)
   - Fixed parameters (dataset, cv) are bound at function construction time

2. **Surrogate evaluation** (use_surrogate=True):
   - Calls through `_evaluate()` -> `_get_surrogate_params()` -> `_surrogate.predict()`
   - `_get_surrogate_params()` adds fixed params (dataset, cv) to hyperparams
   - The surrogate model expects ALL parameters (hyperparams + fixed params)

Data Flow
---------

::

    # Real evaluation path:
    params (hyperparams only)
        -> function.pure_objective_function(params)
        -> actual ML training with bound dataset/cv
        -> real score

    # Surrogate evaluation path:
    params (hyperparams only)
        -> function(params)            # __call__
        -> function._evaluate(params)  # routes based on use_surrogate
        -> function._get_surrogate_params(params)  # adds dataset, cv
        -> function._surrogate.predict(full_params)  # ONNX inference
        -> predicted score

Key Interactions
----------------

- **SurrogateLoader.predict(params)**: Expects dict with ALL parameters
  (hyperparams + fixed params). Iterates over `param_names` from metadata
  to encode values. Will raise KeyError if any expected param is missing.

- **MachineLearningFunction.search_space**: Returns only hyperparameters,
  not fixed parameters. This is what optimizers see.

- **MachineLearningFunction._get_surrogate_params(params)**: Subclasses
  override this to add their fixed parameters (dataset, cv) to the
  hyperparameter dict before calling the surrogate.

Implementation Notes
--------------------

The validator constructor creates a surrogate function instance by copying
all initialization parameters from the real function (dataset, cv, objective,
modifiers, memory, etc.) and setting use_surrogate=True. This ensures both
functions are configured identically except for evaluation mode.

See Also
--------
- surfaces._surrogates._surrogate_loader.SurrogateLoader
- surfaces.test_functions.machine_learning._base_machine_learning.MachineLearningFunction
"""

import inspect
import time
from itertools import product
from typing import Any, Dict, List, Optional

import numpy as np


class SurrogateValidator:
    """Validate surrogate model accuracy against real function evaluations.

    This class compares predictions from a pre-trained surrogate model against
    actual function evaluations to measure approximation quality and speedup.

    The validator creates paired evaluations: for each sampled parameter set,
    it evaluates both the real function (expensive) and the surrogate (fast),
    then computes accuracy metrics and timing comparisons.

    Parameters
    ----------
    function : MachineLearningFunction
        The ML function to validate. Must have ``use_surrogate=False``.
        The validator will create its own surrogate-enabled instance
        internally for comparison.

    Attributes
    ----------
    function : MachineLearningFunction
        The real function instance (use_surrogate=False).
    _surrogate_func : MachineLearningFunction
        The surrogate function instance (use_surrogate=True), created
        internally from the same class.

    Raises
    ------
    ValueError
        If the input function has use_surrogate=True (must be False).
    ValueError
        If no surrogate model is available for the function.

    Examples
    --------
    Basic validation with random sampling:

    >>> from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction
    >>> from surfaces._surrogates import SurrogateValidator
    >>>
    >>> func = KNeighborsClassifierFunction(dataset="iris", cv=5)
    >>> validator = SurrogateValidator(func)
    >>> results = validator.validate_random(n_samples=100)
    >>> print(f"R² = {results['metrics']['r2']:.3f}")
    >>> print(f"Speedup = {results['timing']['speedup']:.0f}x")

    Using grid sampling for systematic coverage:

    >>> results = validator.validate_grid(max_samples=200)

    Running both methods for comprehensive validation:

    >>> summary = validator.summary(n_random=100, n_grid=100)

    Notes
    -----
    **Sampling**: The validator samples from the function's ``search_space``,
    which contains only hyperparameters (not fixed parameters like dataset/cv).
    Both the real and surrogate functions receive the same hyperparameter
    samples.

    **Evaluation paths**:

    - Real: ``pure_objective_function(params)`` - direct ML training
    - Surrogate: ``function(params)`` - routes through ``_evaluate()`` which
      calls ``_get_surrogate_params()`` to add fixed params before querying
      the ONNX model

    **Metrics computed**:

    - R² (coefficient of determination): Overall fit quality, 1.0 = perfect
    - MAE (mean absolute error): Average prediction error magnitude
    - RMSE (root mean squared error): Error with outlier sensitivity
    - Max error: Worst-case prediction error
    - Correlation: Pearson correlation between real and predicted
    - Speedup: Real time / Surrogate time

    See Also
    --------
    SurrogateLoader : Loads and runs ONNX surrogate models.
    MachineLearningFunction : Base class for ML test functions.
    """

    def __init__(self, function):
        """Initialize the validator with a real function instance.

        Creates an internal surrogate-enabled version of the function for
        comparison. Both instances should have the same fixed parameters
        (dataset, cv) to ensure valid comparison.

        Parameters
        ----------
        function : MachineLearningFunction
            The function to validate. Must have ``use_surrogate=False``.

        Raises
        ------
        ValueError
            If function.use_surrogate is True.
        ValueError
            If no surrogate model exists for this function type.

        Notes
        -----
        The surrogate function is created by instantiating the same class
        with all initialization parameters copied from the input function,
        except ``use_surrogate`` which is set to True. This ensures both
        functions use the same fixed parameters (dataset, cv) and configuration
        (objective, modifiers, memory, etc.) for valid comparison.
        """
        self.function = function

        # Ensure we have both real and surrogate versions
        if getattr(function, "use_surrogate", False):
            raise ValueError(
                "Function should have use_surrogate=False. "
                "The validator will create its own surrogate instance."
            )

        # Create surrogate version with same parameters as real function
        func_class = type(function)
        init_params = self._extract_init_params(function)
        init_params["use_surrogate"] = True  # Override to enable surrogate

        self._surrogate_func = func_class(**init_params)

        if not self._surrogate_func.use_surrogate:
            raise ValueError(f"No surrogate model available for {func_class.__name__}")

    def _extract_init_params(self, function) -> Dict[str, Any]:
        """Extract initialization parameters from a function instance.

        Inspects the function's __init__ signature and extracts all parameters
        that have corresponding instance attributes, enabling recreation of
        the function with identical configuration.

        Parameters
        ----------
        function : MachineLearningFunction
            The function instance to extract parameters from.

        Returns
        -------
        dict
            Mapping of parameter names to values, suitable for passing
            to the function's __init__ as **kwargs.

        Notes
        -----
        This method handles:

        - Common parameters: objective, modifiers, memory, collect_data,
          callbacks, catch_errors
        - Function-specific fixed parameters: dataset, cv (tabular),
          epochs, batch_size (image), etc.
        - Skips: self, use_surrogate (will be overridden)

        The extraction is robust to different ML function types (tabular,
        image, timeseries) by using introspection rather than hardcoded lists.
        """
        func_class = type(function)
        sig = inspect.signature(func_class.__init__)

        init_params = {}
        for param_name in sig.parameters:
            # Skip 'self' and 'use_surrogate' (will be overridden)
            if param_name in ("self", "use_surrogate"):
                continue

            # Check if the instance has this attribute
            if hasattr(function, param_name):
                init_params[param_name] = getattr(function, param_name)

        return init_params

    def _get_search_space_values(self) -> Dict[str, List]:
        """Get the search space as a dict of parameter value lists.

        Returns the function's search space, which contains only
        hyperparameters (not fixed parameters like dataset/cv).

        Returns
        -------
        dict
            Mapping of parameter names to lists of possible values.
            Example: {"n_neighbors": [3, 8, 13, ...], "algorithm": ["auto", ...]}
        """
        return self.function.search_space

    def _sample_random(self, n_samples: int, seed: Optional[int] = None) -> List[Dict]:
        """Generate random parameter samples from the search space.

        Samples each parameter independently and uniformly from its
        possible values.

        Parameters
        ----------
        n_samples : int
            Number of random samples to generate.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        list of dict
            List of parameter dictionaries, each containing only
            hyperparameters (not fixed params).
        """
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
        """Generate grid parameter samples from the search space.

        Creates the Cartesian product of all parameter values. If the
        resulting grid is larger than max_samples, randomly subsamples.

        Parameters
        ----------
        max_samples : int, optional
            Maximum number of samples. If the full grid is larger,
            a random subset is selected.

        Returns
        -------
        list of dict
            List of parameter dictionaries covering the grid (or subset).
        """
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
        """Validate surrogate using random parameter samples.

        Generates random samples from the search space and compares
        real vs surrogate evaluations for each.

        Parameters
        ----------
        n_samples : int, default=100
            Number of random samples to evaluate.
        seed : int, optional, default=42
            Random seed for reproducibility.
        verbose : bool, default=True
            If True, print progress and results.

        Returns
        -------
        dict
            Validation results with keys:

            - ``method``: "random"
            - ``n_samples``: Number of successful evaluations
            - ``n_skipped``: Number of failed/NaN evaluations
            - ``metrics``: Dict with r2, mae, rmse, max_error, correlation
            - ``timing``: Dict with avg_real_ms, avg_surrogate_ms, speedup
            - ``data``: Dict with y_real, y_surrogate, errors arrays
        """
        samples = self._sample_random(n_samples, seed)
        return self._validate(samples, "random", verbose)

    def validate_grid(
        self,
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Validate surrogate using grid parameter samples.

        Generates systematic grid samples from the search space and
        compares real vs surrogate evaluations for each.

        Parameters
        ----------
        max_samples : int, optional
            Maximum number of samples. If the full grid is larger,
            a random subset is selected.
        verbose : bool, default=True
            If True, print progress and results.

        Returns
        -------
        dict
            Validation results (same structure as validate_random).
        """
        samples = self._sample_grid(max_samples)
        return self._validate(samples, "grid", verbose)

    def _validate(
        self,
        samples: List[Dict],
        method: str,
        verbose: bool,
    ) -> Dict[str, Any]:
        """Run validation on a list of parameter samples.

        For each sample, evaluates both the real function and the
        surrogate function, collecting predictions and timing.

        Parameters
        ----------
        samples : list of dict
            Parameter samples to evaluate. Each dict contains only
            hyperparameters (the search space parameters).
        method : str
            Sampling method name for reporting ("random" or "grid").
        verbose : bool
            If True, print progress updates.

        Returns
        -------
        dict
            Validation results including metrics, timing, and raw data.

        Notes
        -----
        **Evaluation strategy**:

        - Real evaluation calls ``function.pure_objective_function(params)``
          which directly invokes the ML training with the function's bound
          fixed parameters (dataset, cv).

        - Surrogate evaluation calls ``_surrogate_func(params)`` which:
          1. Routes through ``__call__()`` to ``_evaluate()``
          2. ``_evaluate()`` detects ``use_surrogate=True``
          3. Calls ``_get_surrogate_params(params)`` to add fixed params
          4. Calls ``_surrogate.predict(full_params)`` for ONNX inference

        The samples contain only hyperparameters because that's what the
        search space provides. Fixed parameters are added internally by
        each function based on how it was constructed.

        **Error handling**: If evaluation fails (exception or NaN result),
        the sample is skipped and counted in ``n_skipped``.
        """
        n_samples = len(samples)

        if verbose:
            print(f"Validating {n_samples} samples ({method} sampling)...")

        y_real = []
        y_surr = []
        errors = []
        real_times = []
        surr_times = []

        for i, params in enumerate(samples):
            # Real evaluation: calls the actual ML training function
            # The function uses its bound fixed params (dataset, cv)
            start = time.time()
            try:
                real = self.function.pure_objective_function(params)
                real_time = time.time() - start

                if np.isnan(real):
                    continue

                # Surrogate evaluation: calls through the normal path
                # which routes through _evaluate() -> _get_surrogate_params()
                # This adds the surrogate function's fixed params before
                # calling the ONNX model
                start = time.time()
                surr = self._surrogate_func(params)
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

        # R² score: 1 - (sum of squared residuals / total sum of squares)
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Correlation: Pearson correlation coefficient
        correlation = np.corrcoef(y_real, y_surr)[0, 1]

        # Timing: convert to milliseconds and compute speedup
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
        """Print formatted validation results to stdout.

        Parameters
        ----------
        results : dict
            Validation results from _validate().
        """
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
        """Run both random and grid validation for comprehensive assessment.

        Parameters
        ----------
        n_random : int, default=100
            Number of random samples.
        n_grid : int, default=100
            Maximum grid samples.
        seed : int, default=42
            Random seed for reproducibility.

        Returns
        -------
        dict
            Combined results with keys "random" and "grid", each containing
            the full validation results from the respective method.
        """
        print("Running validation summary...\n")

        random_results = self.validate_random(n_random, seed, verbose=True)
        print()
        grid_results = self.validate_grid(n_grid, verbose=True)

        return {
            "random": random_results,
            "grid": grid_results,
        }
