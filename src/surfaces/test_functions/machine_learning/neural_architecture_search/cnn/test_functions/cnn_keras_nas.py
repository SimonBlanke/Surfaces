"""CNN Neural Architecture Search using Keras/TensorFlow."""

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from surfaces.modifiers import BaseModifier

from ..._base_nas import BaseNeuralArchitectureSearch


class CNNKerasNASFunction(BaseNeuralArchitectureSearch):
    """CNN Neural Architecture Search test function using Keras/TensorFlow.

    **What is optimized:**
    This function optimizes the architecture of a Convolutional Neural Network (CNN)
    for image classification. The search includes:
    - Number of convolutional blocks (1-3)
    - Number of filters per block (16, 32, 64, 128)
    - Kernel size (3x3 or 5x5)
    - Pooling type (MaxPooling or AveragePooling)
    - Number of dense units in fully connected layer (64, 128, 256)

    **Search space structure:**
    The search space is conditional: only the first n_conv_blocks are active.
    For example, if n_conv_blocks=2, only block_1_filters and block_2_filters
    matter; block_3_filters is ignored.

    **What the score means:**
    The score is the validation accuracy (0.0 to 1.0) achieved by the CNN
    architecture on the Fashion-MNIST image classification task after training for
    a fixed number of epochs. Fashion-MNIST contains 28x28 grayscale images across
    10 classes (T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, boot).

    **Optimization goal:**
    MAXIMIZE the validation accuracy. Higher scores indicate architectures that
    better capture visual features and generalize to unseen images. The goal is
    to find the optimal combination of convolutional layers, filter sizes, pooling
    strategies, and dense layer configuration.

    **Computational cost:**
    Each evaluation trains a CNN for multiple epochs on image data, making this
    an expensive function. The default uses a subset of Fashion-MNIST and few epochs
    to keep evaluation time reasonable (~10-30 seconds per evaluation on CPU).

    Parameters
    ----------
    n_epochs : int, default=10
        Number of training epochs per evaluation.
    batch_size : int, default=32
        Training batch size.
    subset_size : int, default=5000
        Number of training samples to use (Fashion-MNIST has 60000 total).
        Smaller values speed up training for prototyping.
    n_jobs : int, default=2
        Number of CPU threads for TensorFlow. Lower values reduce CPU load,
        keeping the system responsive during training.
    objective : str, default="maximize"
        Either "minimize" or "maximize".
    modifiers : list of BaseModifier, optional
        List of modifiers to apply to function evaluations.

    Examples
    --------
    >>> from surfaces.test_functions.machine_learning import CNNKerasNASFunction
    >>> func = CNNKerasNASFunction(n_epochs=5, subset_size=3000)
    >>> func.search_space
    {'n_conv_blocks': [1, 2, 3], 'block_1_filters': [16, 32, 64, 128], ...}
    >>> result = func({"n_conv_blocks": 2, "block_1_filters": 32,
    ...                "block_2_filters": 64, "block_3_filters": 128,
    ...                "kernel_size": 3, "pooling_type": "max",
    ...                "dense_units": 128})
    >>> print(f"Validation accuracy: {result:.4f}")

    Notes
    -----
    Requires TensorFlow. Install with:
        pip install tensorflow

    The function uses Fashion-MNIST, a drop-in replacement for MNIST with more
    challenging image classification. Dataset is ~29MB and cached after first use.
    """

    name = "CNN Keras NAS"
    _name_ = "cnn_keras_nas"
    __name__ = "CNNKerasNASFunction"

    para_names = [
        "n_conv_blocks",
        "block_1_filters",
        "block_2_filters",
        "block_3_filters",
        "kernel_size",
        "pooling_type",
        "dense_units",
    ]
    n_conv_blocks_default = [1, 2, 3]
    block_1_filters_default = [16, 32, 64, 128]
    block_2_filters_default = [16, 32, 64, 128]
    block_3_filters_default = [16, 32, 64, 128]
    kernel_size_default = [3, 5]
    pooling_type_default = ["max", "average"]
    dense_units_default = [64, 128, 256]

    def __init__(
        self,
        n_epochs: int = 10,
        batch_size: int = 32,
        subset_size: int = 5000,
        n_jobs: int = 2,
        objective: str = "maximize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
        use_surrogate: bool = False,
    ):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.subset_size = subset_size
        self.n_jobs = n_jobs

        super().__init__(
            objective=objective,
            modifiers=modifiers,
            memory=memory,
            collect_data=collect_data,
            callbacks=callbacks,
            catch_errors=catch_errors,
            use_surrogate=use_surrogate,
        )

    @property
    def search_space(self) -> Dict[str, Any]:
        """Search space for CNN architecture optimization."""
        return {
            "n_conv_blocks": self.n_conv_blocks_default,
            "block_1_filters": self.block_1_filters_default,
            "block_2_filters": self.block_2_filters_default,
            "block_3_filters": self.block_3_filters_default,
            "kernel_size": self.kernel_size_default,
            "pooling_type": self.pooling_type_default,
            "dense_units": self.dense_units_default,
        }

    def _create_objective_function(self) -> None:
        """Create objective function for CNN NAS."""
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        # Limit CPU threads to keep system responsive
        tf.config.threading.set_inter_op_parallelism_threads(self.n_jobs)
        tf.config.threading.set_intra_op_parallelism_threads(self.n_jobs)

        # Load Fashion-MNIST dataset
        (X_train, y_train), (X_val, y_val) = keras.datasets.fashion_mnist.load_data()

        # Normalize to [0, 1] and add channel dimension
        X_train = X_train.astype("float32") / 255.0
        X_val = X_val.astype("float32") / 255.0
        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]

        # Use subset for faster evaluation
        if self.subset_size < len(X_train):
            indices = np.random.RandomState(42).choice(
                len(X_train), self.subset_size, replace=False
            )
            X_train = X_train[indices]
            y_train = y_train[indices]

        # Take smaller validation set too
        val_size = min(2000, len(X_val))
        X_val = X_val[:val_size]
        y_val = y_val[:val_size]

        n_epochs = self.n_epochs
        batch_size = self.batch_size

        def objective_function(params: Dict[str, Any]) -> float:
            # Build CNN architecture
            n_conv_blocks = params["n_conv_blocks"]
            kernel_size = params["kernel_size"]
            pooling_type = params["pooling_type"]
            dense_units = params["dense_units"]

            model = keras.Sequential()
            model.add(layers.Input(shape=(28, 28, 1)))

            # Add convolutional blocks
            for i in range(1, n_conv_blocks + 1):
                filters = params[f"block_{i}_filters"]
                model.add(
                    layers.Conv2D(
                        filters, kernel_size=kernel_size, activation="relu", padding="same"
                    )
                )
                if pooling_type == "max":
                    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
                else:  # average
                    model.add(layers.AveragePooling2D(pool_size=(2, 2)))

            # Flatten and dense layers
            model.add(layers.Flatten())
            model.add(layers.Dense(dense_units, activation="relu"))
            model.add(layers.Dense(10, activation="softmax"))

            # Compile model
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            # Train model (suppress output)
            history = model.fit(
                X_train,
                y_train,
                batch_size=batch_size,
                epochs=n_epochs,
                validation_data=(X_val, y_val),
                verbose=0,
            )

            # Return final validation accuracy
            val_accuracy = history.history["val_accuracy"][-1]
            return val_accuracy

        self.pure_objective_function = objective_function
