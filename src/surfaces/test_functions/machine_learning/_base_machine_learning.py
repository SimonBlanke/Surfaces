# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import functools
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from surfaces.modifiers import BaseModifier

from ..._surrogates import load_surrogate
from .._base_single_objective import BaseSingleObjectiveTestFunction


def _load_surrogate_before_init(init_func):
    """Ensure surrogate loading happens before __init__ body executes.

    This guarantees that ``self.use_surrogate`` and ``self._surrogate``
    are set before ``super().__init__()`` triggers
    ``_check_dependencies()``, so the ML override can skip the check
    when a surrogate is active.
    """

    @functools.wraps(init_func)
    def wrapper(self, *args, **kwargs):
        use_surrogate = kwargs.get("use_surrogate", False)
        self.use_surrogate = use_surrogate
        self._surrogate = None
        if use_surrogate:
            self._load_surrogate()
        return init_func(self, *args, **kwargs)

    return wrapper


class MachineLearningFunction(BaseSingleObjectiveTestFunction):
    """
    Base class for machine learning hyperparameter optimization test functions.

    ML functions evaluate model performance based on hyperparameter configurations.
    They naturally return score values where higher is better.

    Parameters
    ----------
    objective : str, default="maximize"
        Either "minimize" or "maximize".
    modifiers : list of BaseModifier, optional
        List of modifiers to apply to function evaluations.
    use_surrogate : bool, default=False
        If True and a pre-trained surrogate exists, use it for fast evaluation.
        Falls back to real evaluation if no surrogate is available.
    """

    _spec = {
        "continuous": False,
        "differentiable": False,
        "stochastic": True,
    }

    para_names: list = []

    def _default_search_space(self) -> Dict[str, Any]:
        """Build search space from *_default class attributes."""
        search_space = {}
        for param_name in self.para_names:
            default_attr = f"{param_name}_default"
            if hasattr(self, default_attr):
                search_space[param_name] = getattr(self, default_attr)
        return search_space

    @_load_surrogate_before_init
    def __init__(
        self,
        objective: str = "maximize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
        use_surrogate: bool = False,
        **kwargs: Any,
    ) -> None:
        # use_surrogate / _surrogate already set by @_load_surrogate_before_init
        super().__init__(objective, modifiers, memory, collect_data, callbacks, catch_errors)

    def _load_surrogate(self) -> None:
        """Load pre-trained surrogate model if available."""
        function_name = getattr(self, "_name_", self.__class__.__name__)
        self._surrogate = load_surrogate(function_name)

        if self._surrogate is None:
            import warnings

            warnings.warn(
                f"No surrogate model found for '{function_name}'. Falling back to real evaluation.",
                UserWarning,
            )
            self.use_surrogate = False
            return

        stored_fp = self._surrogate.interface_fingerprint
        if stored_fp is not None:
            from ..._surrogates._surrogate_loader import compute_interface_fingerprint

            current_fp = compute_interface_fingerprint(self)
            if current_fp != stored_fp:
                import warnings

                warnings.warn(
                    f"Surrogate interface mismatch for '{function_name}'. "
                    f"The function's interface has changed since the surrogate was trained "
                    f"(expected {stored_fp}, got {current_fp}). "
                    f"Predictions may be inaccurate. Retrain the surrogate.",
                    UserWarning,
                )

    def _check_dependencies(self):
        """Skip dependency check when using a surrogate model."""
        if self.use_surrogate and self._surrogate is not None:
            return
        super()._check_dependencies()

    def _ml_objective(self, params: Dict[str, Any]) -> float:
        """Compute the ML objective value for given hyperparameters.

        Override in subclasses to define the ML training/evaluation logic.

        Parameters
        ----------
        params : dict
            Hyperparameter values.

        Returns
        -------
        float
            Score value (higher is better, e.g. accuracy).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _ml_objective(self, params)"
        )

    def _objective(self, params: Dict[str, Any]) -> float:
        """Sub-template: delegates to _ml_objective."""
        return self._ml_objective(params)

    def _get_surrogate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameters for surrogate prediction.

        Override in subclasses to add fixed parameters (like dataset, cv)
        that are not in the search space but needed by the surrogate.

        Parameters
        ----------
        params : dict
            Search parameters from the optimizer.

        Returns
        -------
        dict
            Full parameters for surrogate prediction.
        """
        return params

    def _apply_direction(self, value):
        """ML functions return higher-is-better, so negate for minimize."""
        if self.objective == "minimize":
            return -value
        return value

    def _get_training_data(self):
        """Load training data with optional fidelity-based subsampling.

        Reads ``self._active_fidelity`` (set by ``_evaluate`` during the
        current call). When fidelity is None or 1.0 the full dataset is
        returned. Otherwise a fraction of the data is selected.

        Subclasses can override ``_subsample_data`` to control the
        subsampling strategy (stratified for classification, sequential
        for time-series, etc.).

        Raises
        ------
        ValueError
            If the subsampled data would have fewer samples than required
            for cross-validation (controlled by ``self.cv`` if present).
        """
        X, y = self._dataset_loader()
        fidelity = getattr(self, "_active_fidelity", None)
        if fidelity is not None and fidelity < 1.0:
            X, y = self._subsample_data(X, y, fidelity)
            min_samples = getattr(self, "cv", 2)
            if len(X) < min_samples:
                raise ValueError(
                    f"fidelity={fidelity} produces only {len(X)} samples, "
                    f"but at least {min_samples} are required (cv={min_samples}). "
                    f"Use a higher fidelity value."
                )
        return X, y

    def _subsample_data(self, X, y, fidelity: float):
        """Select a fraction of the training data.

        Default: deterministic shuffled sampling. Override in subclasses
        for task-specific strategies (stratified, sequential, etc.).
        """
        n_samples = max(1, int(len(X) * fidelity))
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(X))[:n_samples]
        return X[indices], y[indices]

    def _evaluate(self, params: Dict[str, Any], fidelity: Optional[float] = None) -> float:
        """Evaluate with surrogate support, fidelity, and direction handling.

        ML functions naturally return scores (higher is better),
        so direction is inverted compared to algebraic functions.
        """
        self._active_fidelity = fidelity
        try:
            if self.use_surrogate and self._surrogate is not None:
                surrogate_params = self._get_surrogate_params(params)
                if self._surrogate.fidelity_aware:
                    raw_value = self._surrogate.predict(surrogate_params, fidelity=fidelity)
                else:
                    if fidelity is not None and fidelity < 1.0:
                        import warnings

                        warnings.warn(
                            f"fidelity={fidelity} has no effect with this surrogate model. "
                            "Retrain the surrogate with fidelity support to enable "
                            "multi-fidelity predictions.",
                            UserWarning,
                            stacklevel=3,
                        )
                    raw_value = self._surrogate.predict(surrogate_params)
            else:
                raw_value = self._objective(params)

            if self._modifiers:
                raw_value = self._apply_modifiers(raw_value, params)

            return self._apply_direction(raw_value)
        finally:
            self._active_fidelity = None
