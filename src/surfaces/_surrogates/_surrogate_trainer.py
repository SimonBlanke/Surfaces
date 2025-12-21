# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Private interface for training surrogate models.

This module is intended for package maintainers to create
pre-trained surrogate models for expensive functions.
"""

import json
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class SurrogateTrainer:
    """Train surrogate models for expensive objective functions.

    This is a private interface for package maintainers to create
    pre-trained ONNX surrogate models.

    Parameters
    ----------
    function : BaseTestFunction
        The expensive function to approximate.
    output_dir : Path or str, optional
        Directory to save trained models. Defaults to the models/ directory.

    Examples
    --------
    >>> from surfaces import KNeighborsClassifierFunction
    >>> func = KNeighborsClassifierFunction()
    >>> trainer = SurrogateTrainer(func)
    >>> trainer.collect_samples(n_samples=1000)
    >>> trainer.train()
    >>> trainer.export("k_neighbors_classifier.onnx")
    """

    def __init__(
        self,
        function,
        output_dir: Optional[Path] = None,
    ):
        self.function = function
        self.output_dir = Path(output_dir) if output_dir else (Path(__file__).parent / "models")

        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.X_all: Optional[np.ndarray] = None  # All samples (valid + invalid)
        self.y_valid: Optional[np.ndarray] = None  # Validity labels (0/1)
        self.param_names: List[str] = []
        self.param_encodings: Dict[str, Dict[str, int]] = {}
        self.model = None
        self.validity_model = None

        self._training_time: float = 0
        self._collection_time: float = 0

    def _is_numeric(self, value) -> bool:
        """Check if a value is numeric (including numpy types)."""
        if isinstance(value, (int, float)):
            return True
        if hasattr(value, "dtype"):
            return np.issubdtype(value.dtype, np.number)
        return False

    def _encode_search_space(self) -> Tuple[List[str], Dict, List[List]]:
        """Analyze search space and create encodings for categorical params.

        Returns
        -------
        param_names : list
            Ordered parameter names.
        encodings : dict
            Mapping of categorical param names to {value: int} dicts.
        numeric_spaces : list
            List of numeric values for each parameter.
        """
        search_space = self.function.search_space
        param_names = sorted(search_space.keys())
        encodings = {}
        numeric_spaces = []

        for name in param_names:
            values = search_space[name]

            # Check if values are numeric or categorical
            if all(self._is_numeric(v) for v in values):
                numeric_spaces.append(values)
            else:
                # Categorical: create encoding
                encoding = {}
                for i, v in enumerate(values):
                    if callable(v):
                        key = v.__name__
                    else:
                        key = str(v)
                    encoding[key] = i
                encodings[name] = encoding
                numeric_spaces.append(list(range(len(values))))

        return param_names, encodings, numeric_spaces

    def collect_samples_grid(
        self,
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collect samples using full grid search.

        Parameters
        ----------
        max_samples : int, optional
            Maximum number of samples. If grid is larger, subsample randomly.
        verbose : bool
            Print progress information.

        Returns
        -------
        X : ndarray
            Input features (n_samples, n_params).
        y : ndarray
            Target values (n_samples,).
        """
        start_time = time.time()

        param_names, encodings, numeric_spaces = self._encode_search_space()
        self.param_names = param_names
        self.param_encodings = encodings

        search_space = self.function.search_space

        # Generate all grid points
        grid_points = list(product(*[search_space[name] for name in param_names]))
        n_total = len(grid_points)

        if max_samples and n_total > max_samples:
            if verbose:
                print(f"Grid has {n_total} points, subsampling to {max_samples}")
            indices = np.random.choice(n_total, max_samples, replace=False)
            grid_points = [grid_points[i] for i in indices]

        n_samples = len(grid_points)
        X_valid_list = []
        y_list = []
        X_all_list = []
        validity_list = []

        if verbose:
            print(f"Collecting {n_samples} samples...")

        for i, point in enumerate(grid_points):
            params = {name: val for name, val in zip(param_names, point)}

            # Encode for X
            x_row = []
            for name, val in zip(param_names, point):
                if name in encodings:
                    if callable(val):
                        key = val.__name__
                    else:
                        key = str(val)
                    x_row.append(float(encodings[name][key]))
                else:
                    x_row.append(float(val))

            # Evaluate function (use pure_objective_function to get raw value)
            try:
                score = self.function.pure_objective_function(params)

                # Track all samples for validity model
                X_all_list.append(x_row)

                if np.isnan(score):
                    # Invalid combination
                    validity_list.append(0)
                else:
                    # Valid combination
                    validity_list.append(1)
                    X_valid_list.append(x_row)
                    y_list.append(score)

                if verbose and (i + 1) % 100 == 0:
                    print(f"  Collected {len(y_list)}/{n_samples} valid samples")
            except Exception as e:
                # Treat exceptions as invalid
                X_all_list.append(x_row)
                validity_list.append(0)
                if verbose:
                    print(f"  Error at sample {i}: {e}")

        self.X = np.array(X_valid_list, dtype=np.float32)
        self.y = np.array(y_list, dtype=np.float32)
        self.X_all = np.array(X_all_list, dtype=np.float32)
        self.y_valid = np.array(validity_list, dtype=np.int32)

        self._collection_time = time.time() - start_time

        n_valid = len(self.y)
        n_invalid = len(self.y_valid) - n_valid

        if verbose:
            print(f"Collected {n_valid} valid samples in {self._collection_time:.1f}s")
            if n_invalid > 0:
                print(f"  Invalid samples: {n_invalid} (will train validity model)")
            if n_valid > 0:
                print(f"  y range: [{self.y.min():.4f}, {self.y.max():.4f}]")

        return self.X, self.y

    def train(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (64, 64),
        max_iter: int = 1000,
        verbose: bool = True,
    ):
        """Train an MLP regressor on collected samples.

        Also trains a validity classifier if invalid samples were found.

        Parameters
        ----------
        hidden_layer_sizes : tuple
            Size of hidden layers.
        max_iter : int
            Maximum training iterations.
        verbose : bool
            Print training progress.
        """
        if self.X is None or self.y is None:
            raise ValueError("No samples collected. Call collect_samples_grid first.")

        try:
            from sklearn.neural_network import MLPRegressor
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("sklearn is required for training surrogates")

        start_time = time.time()

        # Normalize inputs for regression model
        self.scaler_X = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(self.X)

        # Train regression MLP
        if verbose:
            print("Training regression model...")
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            verbose=verbose,
        )
        self.model.fit(X_scaled, self.y)

        # Evaluate regression on training data
        y_pred = self.model.predict(X_scaled)
        mse = np.mean((self.y - y_pred) ** 2)
        r2 = 1 - mse / np.var(self.y)

        # Train validity classifier if there are invalid samples
        n_invalid = np.sum(self.y_valid == 0)
        if n_invalid > 0:
            if verbose:
                print("\nTraining validity classifier (DecisionTree)...")

            from sklearn.tree import DecisionTreeClassifier

            # Decision tree doesn't need scaling, but we keep scaler for API consistency
            self.scaler_X_validity = None

            self.validity_model = DecisionTreeClassifier(
                max_depth=10,
                min_samples_leaf=5,
                random_state=42,
            )
            self.validity_model.fit(self.X_all, self.y_valid)

            # Evaluate validity classifier
            validity_pred = self.validity_model.predict(self.X_all)
            validity_acc = np.mean(validity_pred == self.y_valid)

            if verbose:
                print(f"  Validity classifier accuracy: {validity_acc:.4f}")
                print(f"  Tree depth: {self.validity_model.get_depth()}")

        self._training_time = time.time() - start_time

        if verbose:
            print(f"\nTraining completed in {self._training_time:.1f}s")
            print(f"  Regression MSE: {mse:.6f}")
            print(f"  Regression R2:  {r2:.4f}")

    def export(
        self,
        filename: str,
        verbose: bool = True,
    ) -> Path:
        """Export trained model to ONNX format.

        Parameters
        ----------
        filename : str
            Output filename (e.g., "k_neighbors_classifier.onnx").
        verbose : bool
            Print export information.

        Returns
        -------
        Path
            Path to the exported ONNX file.
        """
        if self.model is None:
            raise ValueError("No model trained. Call train() first.")

        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError:
            raise ImportError(
                "skl2onnx is required for ONNX export. Install it with: pip install skl2onnx"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / filename
        metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")

        # Create a wrapper pipeline that includes scaling
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline(
            [
                ("scaler", self.scaler_X),
                ("mlp", self.model),
            ]
        )

        # Convert regression model to ONNX
        n_features = self.X.shape[1]
        initial_type = [("input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(pipeline, initial_types=initial_type)

        # Save ONNX model
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        # Export validity model if it exists
        has_validity_model = self.validity_model is not None
        if has_validity_model:
            validity_path = output_path.with_suffix(".validity.onnx")

            # DecisionTree doesn't need a scaler pipeline
            onnx_validity = convert_sklearn(
                self.validity_model,
                initial_types=initial_type,
                options={id(self.validity_model): {"zipmap": False}},
            )

            with open(validity_path, "wb") as f:
                f.write(onnx_validity.SerializeToString())

            if verbose:
                print(f"Exported validity model to: {validity_path}")

        # Save metadata
        n_invalid = int(np.sum(self.y_valid == 0))
        metadata = {
            "function_name": getattr(self.function, "_name_", self.function.__class__.__name__),
            "param_names": self.param_names,
            "param_encodings": self.param_encodings,
            "n_samples": len(self.y),
            "n_invalid_samples": n_invalid,
            "has_validity_model": has_validity_model,
            "y_range": [float(self.y.min()), float(self.y.max())],
            "training_time": self._training_time,
            "collection_time": self._collection_time,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        if verbose:
            print(f"Exported model to: {output_path}")
            print(f"Exported metadata to: {metadata_path}")

        return output_path


def train_surrogate_for_function(
    function,
    output_name: str,
    max_samples: int = 5000,
    hidden_layers: Tuple[int, ...] = (64, 64),
    verbose: bool = True,
) -> Path:
    """Convenience function to train and export a surrogate.

    Parameters
    ----------
    function : BaseTestFunction
        The function to create a surrogate for.
    output_name : str
        Name for the output file (without .onnx extension).
    max_samples : int
        Maximum training samples.
    hidden_layers : tuple
        MLP hidden layer sizes.
    verbose : bool
        Print progress.

    Returns
    -------
    Path
        Path to the exported ONNX file.
    """
    trainer = SurrogateTrainer(function)
    trainer.collect_samples_grid(max_samples=max_samples, verbose=verbose)
    trainer.train(hidden_layer_sizes=hidden_layers, verbose=verbose)
    return trainer.export(f"{output_name}.onnx", verbose=verbose)
