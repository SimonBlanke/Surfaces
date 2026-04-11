# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Analysis namespace for CustomTestFunction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import numpy as np

if TYPE_CHECKING:
    from .._custom_test_function import CustomTestFunction


class AnalysisNamespace:
    """Analysis tools for understanding optimization results.

    This namespace provides methods for analyzing the collected evaluation
    data, including parameter importance, convergence detection, and
    landscape characterization.

    Parameters
    ----------
    func : CustomTestFunction
        The parent function to analyze.

    Examples
    --------
    >>> func.analysis.summary()
    >>> importance = func.analysis.parameter_importance()
    >>> func.analysis.convergence()
    """

    def __init__(self, func: "CustomTestFunction") -> None:
        self._func = func

    def _check_data(self, min_evaluations: int = 1) -> None:
        """Check that sufficient data is available."""
        if self._func.n_evaluations < min_evaluations:
            raise ValueError(
                f"Analysis requires at least {min_evaluations} evaluations, "
                f"got {self._func.n_evaluations}"
            )

    def summary(self) -> Dict[str, Any]:
        """Generate a summary of the optimization results.

        Returns
        -------
        dict
            Summary statistics including:
            - n_evaluations: Number of evaluations
            - best_score: Best score found
            - best_params: Best parameters
            - total_time: Total evaluation time
            - mean_score: Mean score
            - std_score: Standard deviation of scores

        Examples
        --------
        >>> summary = func.analysis.summary()
        >>> print(f"Best score: {summary['best_score']}")
        """
        self._check_data()

        X, y = self._func.get_data_as_arrays()

        return {
            "n_evaluations": self._func.n_evaluations,
            "n_dim": self._func.n_dim,
            "param_names": self._func.param_names,
            "best_score": self._func.best_score,
            "best_params": self._func.best_params,
            "total_time": self._func.total_time,
            "mean_score": float(np.mean(y)),
            "std_score": float(np.std(y)),
            "min_score": float(np.min(y)),
            "max_score": float(np.max(y)),
            "experiment": self._func.experiment,
            "tags": self._func.tags,
        }

    def parameter_importance(self, method: str = "variance") -> Dict[str, float]:
        """Calculate parameter importance scores.

        Parameters
        ----------
        method : str, default="variance"
            Method for calculating importance:
            - "variance": Variance-based importance (fast, approximate)
            - "fanova": Functional ANOVA (requires more data, more accurate)

        Returns
        -------
        dict
            Mapping of parameter names to importance scores (sum to 1.0).

        Examples
        --------
        >>> importance = func.analysis.parameter_importance()
        >>> print(importance)
        {'learning_rate': 0.65, 'n_estimators': 0.25, 'max_depth': 0.10}
        """
        self._check_data(min_evaluations=10)

        if method == "variance":
            return self._variance_importance()
        elif method == "fanova":
            return self._fanova_importance()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'variance' or 'fanova'.")

    def _variance_importance(self) -> Dict[str, float]:
        """Calculate variance-based parameter importance."""
        X, y = self._func.get_data_as_arrays()
        param_names = self._func.param_names

        importances = {}
        for i, name in enumerate(param_names):
            # Correlation between parameter and score
            correlation = np.abs(np.corrcoef(X[:, i], y)[0, 1])
            if np.isnan(correlation):
                correlation = 0.0
            importances[name] = correlation

        # Normalize to sum to 1
        total = sum(importances.values())
        if total > 0:
            importances = {k: v / total for k, v in importances.items()}

        return importances

    def _fanova_importance(self) -> Dict[str, float]:
        """Calculate fANOVA-based parameter importance.

        Fits a Random Forest to the collected data, then decomposes
        the predicted variance into per-parameter contributions by
        analytically marginalizing over all other parameters in each
        tree's leaf structure. Captures non-linear effects that the
        correlation-based variance method misses.
        """
        from sklearn.ensemble import RandomForestRegressor

        self._check_data(min_evaluations=30)

        X, y = self._func.get_data_as_arrays()
        param_names = self._func.param_names
        bounds_dict = self._func.bounds
        n_dims = len(param_names)

        global_bounds = [list(bounds_dict[name]) for name in param_names]

        rf = RandomForestRegressor(n_estimators=64, random_state=42)
        rf.fit(X, y)

        def get_leaves(tree, node, bounds):
            """Recursively extract leaf predictions and their hyperrectangle bounds."""
            if tree.feature[node] < 0:
                return [(tree.value[node].item(), [list(b) for b in bounds])]

            feat = tree.feature[node]
            thresh = tree.threshold[node]

            left_b = [list(b) for b in bounds]
            left_b[feat] = [left_b[feat][0], min(left_b[feat][1], thresh)]

            right_b = [list(b) for b in bounds]
            right_b[feat] = [max(right_b[feat][0], thresh), right_b[feat][1]]

            return get_leaves(tree, tree.children_left[node], left_b) + get_leaves(
                tree, tree.children_right[node], right_b
            )

        def marginal_variance(leaves, dim):
            """Variance of the marginal prediction along one dimension.

            Integrates out all other dimensions analytically using the
            piecewise-constant structure of the tree, then computes the
            variance of the resulting one-dimensional step function.
            """
            g_lo, g_hi = global_bounds[dim]
            total_range = g_hi - g_lo
            if total_range <= 0:
                return 0.0

            cut_points = {g_lo, g_hi}
            for _, bds in leaves:
                cut_points.add(bds[dim][0])
                cut_points.add(bds[dim][1])
            cut_points = sorted(cut_points)

            means = []
            weights = []
            for j in range(len(cut_points) - 1):
                lo, hi = cut_points[j], cut_points[j + 1]
                if hi <= lo:
                    continue

                pred = 0.0
                for val, bds in leaves:
                    if bds[dim][0] > lo or bds[dim][1] < hi:
                        continue
                    w = 1.0
                    for d in range(n_dims):
                        if d == dim:
                            continue
                        r = global_bounds[d][1] - global_bounds[d][0]
                        if r > 0:
                            w *= (bds[d][1] - bds[d][0]) / r
                    pred += val * w

                means.append(pred)
                weights.append((hi - lo) / total_range)

            if not means:
                return 0.0

            mu = sum(m * w for m, w in zip(means, weights))
            return sum((m - mu) ** 2 * w for m, w in zip(means, weights))

        importances = np.zeros(n_dims)
        for estimator in rf.estimators_:
            tree = estimator.tree_
            leaves = get_leaves(tree, 0, [list(b) for b in global_bounds])
            for i in range(n_dims):
                importances[i] += marginal_variance(leaves, i)

        importances /= len(rf.estimators_)

        total = importances.sum()
        if total > 0:
            importances /= total
        else:
            importances = np.full(n_dims, 1.0 / n_dims)

        return {name: float(importances[i]) for i, name in enumerate(param_names)}

    def convergence(self) -> Dict[str, Any]:
        """Analyze optimization convergence.

        Returns
        -------
        dict
            Convergence analysis including:
            - is_converged: Whether optimization appears converged
            - best_at_eval: Evaluation number where best was found
            - improvement_rate: Rate of improvement over evaluations

        Examples
        --------
        >>> conv = func.analysis.convergence()
        >>> if conv['is_converged']:
        ...     print("Optimization has converged")
        """
        self._check_data(min_evaluations=5)

        X, y = self._func.get_data_as_arrays()

        # Find when best was achieved
        best_idx = np.argmin(y) if self._func.objective == "minimize" else np.argmax(y)

        # Check if improvement in last 20% of evaluations
        n = len(y)
        last_20_pct = int(n * 0.8)
        recent_best_idx = (
            np.argmin(y[last_20_pct:])
            if self._func.objective == "minimize"
            else np.argmax(y[last_20_pct:])
        )
        improved_recently = recent_best_idx > 0

        # Calculate running best
        if self._func.objective == "minimize":
            running_best = np.minimum.accumulate(y)
        else:
            running_best = np.maximum.accumulate(y)

        return {
            "is_converged": not improved_recently and best_idx < last_20_pct,
            "best_at_eval": int(best_idx) + 1,
            "best_in_last_20_pct": improved_recently,
            "running_best": running_best.tolist(),
        }

    def suggest_refined_space(self, quantile: float = 0.1) -> Dict[str, tuple]:
        """Suggest a refined search space based on top evaluations.

        Parameters
        ----------
        quantile : float, default=0.1
            Use top quantile of evaluations to determine refined bounds.

        Returns
        -------
        dict
            Refined search space as {param: (min, max)} dict.

        Examples
        --------
        >>> refined = func.analysis.suggest_refined_space()
        >>> # Use refined space for next optimization round
        """
        self._check_data(min_evaluations=10)

        X, y = self._func.get_data_as_arrays()
        param_names = self._func.param_names

        # Find top evaluations
        n_top = max(1, int(len(y) * quantile))
        if self._func.objective == "minimize":
            top_indices = np.argsort(y)[:n_top]
        else:
            top_indices = np.argsort(y)[-n_top:]

        X_top = X[top_indices]

        # Calculate refined bounds
        refined = {}
        for i, name in enumerate(param_names):
            values = X_top[:, i]
            margin = (values.max() - values.min()) * 0.1
            refined[name] = (
                float(values.min() - margin),
                float(values.max() + margin),
            )

        return refined
