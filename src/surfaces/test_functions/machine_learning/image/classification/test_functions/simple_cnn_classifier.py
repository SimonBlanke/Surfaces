from typing import Any, Dict

# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Simple CNN Image Classifier test function."""

import numpy as np

from .._base_image_classification import BaseImageClassification
from ..datasets import DATASETS


def _check_tensorflow():
    """Check if tensorflow is available."""
    try:
        import tensorflow  # noqa: F401

        return True
    except ImportError:
        raise ImportError(
            "CNN image classifiers require tensorflow. "
            "Install with: pip install surfaces[images]"
        )


class SimpleCNNClassifierFunction(BaseImageClassification):
    """Simple CNN Image Classifier test function.

    A basic Convolutional Neural Network for image classification.
    Requires tensorflow/keras.

    Parameters
    ----------
    dataset : str, default="mnist"
        Dataset to use. One of: "mnist", "fashion_mnist".
    epochs : int, default=5
        Number of training epochs.
    validation_split : float, default=0.2
        Fraction of data used for validation.
    use_surrogate : bool, default=False
        If True, use pre-trained surrogate for fast evaluation.
    objective : str, default="maximize"
        Either "minimize" or "maximize".
    sleep : float, default=0
        Artificial delay in seconds.

    Examples
    --------
    >>> from surfaces.test_functions.machine_learning.image import (
    ...     SimpleCNNClassifierFunction
    ... )
    >>> func = SimpleCNNClassifierFunction(dataset="mnist", epochs=3)
    >>> result = func({"filters": 32, "kernel_size": 3, "dense_units": 64})
    """

    name = "Simple CNN Classifier Function"
    _name_ = "simple_cnn_classifier"
    __name__ = "SimpleCNNClassifierFunction"

    available_datasets = list(DATASETS.keys())

    # Search space parameters
    para_names = ["filters", "kernel_size", "dense_units"]
    filters_default = [16, 32, 64, 128]
    kernel_size_default = [3, 5, 7]
    dense_units_default = [32, 64, 128, 256]

    def __init__(
        self,
        dataset: str = "mnist",
        epochs: int = 5,
        validation_split: float = 0.2,
        objective: str = "maximize",
        sleep: float = 0,
        memory: bool = False,
        collect_data: bool = True,
        callbacks=None,
        catch_errors=None,
        noise=None,
        use_surrogate: bool = False,
    ):
        _check_tensorflow()

        if dataset not in DATASETS:
            raise ValueError(
                f"Unknown dataset '{dataset}'. " f"Available: {self.available_datasets}"
            )

        self.dataset = dataset
        self.epochs = epochs
        self.validation_split = validation_split
        self._dataset_loader = DATASETS[dataset]

        super().__init__(
            objective=objective,
            sleep=sleep,
            memory=memory,
            collect_data=collect_data,
            callbacks=callbacks,
            catch_errors=catch_errors,
            noise=noise,
            use_surrogate=use_surrogate,
        )

    @property
    def search_space(self) -> Dict[str, Any]:
        """Search space containing hyperparameters."""
        return {
            "filters": self.filters_default,
            "kernel_size": self.kernel_size_default,
            "dense_units": self.dense_units_default,
        }

    def _create_objective_function(self) -> None:
        """Create objective function with fixed dataset and epochs."""
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        # Suppress TF warnings
        tf.get_logger().setLevel("ERROR")

        X_raw, y = self._dataset_loader()

        # Reshape for CNN (samples, height, width, channels)
        img_size = int(np.sqrt(X_raw.shape[1]))
        X = X_raw.reshape(-1, img_size, img_size, 1).astype("float32")

        # Normalize to [0, 1]
        X = X / X.max()

        n_classes = len(np.unique(y))
        epochs = self.epochs
        validation_split = self.validation_split

        def simple_cnn_classifier(params):
            # Clear session to avoid memory accumulation
            keras.backend.clear_session()

            model = keras.Sequential(
                [
                    layers.Conv2D(
                        params["filters"],
                        (params["kernel_size"], params["kernel_size"]),
                        activation="relu",
                        input_shape=(img_size, img_size, 1),
                        padding="same",
                    ),
                    layers.MaxPooling2D((2, 2)),
                    layers.Conv2D(
                        params["filters"] * 2,
                        (params["kernel_size"], params["kernel_size"]),
                        activation="relu",
                        padding="same",
                    ),
                    layers.MaxPooling2D((2, 2)),
                    layers.Flatten(),
                    layers.Dense(params["dense_units"], activation="relu"),
                    layers.Dropout(0.5),
                    layers.Dense(n_classes, activation="softmax"),
                ]
            )

            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            history = model.fit(
                X,
                y,
                epochs=epochs,
                validation_split=validation_split,
                verbose=0,
                batch_size=32,
            )

            # Return best validation accuracy
            return max(history.history["val_accuracy"])

        self.pure_objective_function = simple_cnn_classifier

    def _get_surrogate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add fixed parameters for surrogate prediction."""
        return {
            **params,
            "dataset": self.dataset,
            "epochs": self.epochs,
        }
