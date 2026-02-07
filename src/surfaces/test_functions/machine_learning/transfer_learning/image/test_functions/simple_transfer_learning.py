"""Simple Transfer Learning: MNIST to Fashion-MNIST."""

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from surfaces.modifiers import BaseModifier

from ..._base_transfer_learning import BaseTransferLearning


class SimpleTransferLearningFunction(BaseTransferLearning):
    """Transfer Learning test function: pretrain on MNIST, fine-tune on Fashion-MNIST.

    **What is optimized:**
    This function optimizes the transfer learning configuration when adapting
    a CNN pretrained on MNIST (handwritten digits) to Fashion-MNIST (clothing items).
    The search includes:
    - Freeze ratio (0.0-1.0): fraction of pretrained layers to freeze
    - Fine-tuning learning rate (1e-5, 1e-4, 1e-3)
    - Dropout rate for regularization (0.0, 0.2, 0.4)
    - Dense layer size before output (32, 64, 128)

    **Transfer learning concept:**
    Transfer learning leverages knowledge from a model pretrained on one task (MNIST)
    and adapts it to a related task (Fashion-MNIST). Both datasets share the same
    image format (28x28 grayscale) but differ in content. Freezing early layers
    preserves learned low-level features (edges, textures) while allowing later
    layers to adapt to the new task.

    **What the score means:**
    The score is the validation accuracy (0.0 to 1.0) achieved after fine-tuning
    the pretrained model on Fashion-MNIST. Higher scores indicate better transfer
    of knowledge from digit recognition to clothing classification.

    **Optimization goal:**
    MAXIMIZE the validation accuracy. The goal is to find the optimal balance between:
    - Freezing layers (preserving MNIST features vs. task-specific adaptation)
    - Learning rate (fast convergence vs. catastrophic forgetting)
    - Regularization (preventing overfitting)
    - Model capacity (dense layer size)

    **Computational cost:**
    Each evaluation involves pretraining on MNIST and fine-tuning on Fashion-MNIST.
    The default uses small subsets and few epochs to keep evaluation time reasonable
    (~10-30 seconds per evaluation on CPU).

    **Why this design:**
    This function uses only native Keras datasets (MNIST, Fashion-MNIST) that are
    small (~11MB and ~29MB) and require no external downloads beyond the initial
    cache. This makes development and testing fast and reliable.

    Parameters
    ----------
    pretrain_epochs : int, default=3
        Number of epochs to pretrain on MNIST.
    finetune_epochs : int, default=5
        Number of epochs to fine-tune on Fashion-MNIST.
    batch_size : int, default=64
        Training batch size.
    pretrain_subset : int, default=5000
        Number of MNIST samples for pretraining.
    finetune_subset : int, default=5000
        Number of Fashion-MNIST samples for fine-tuning.
    n_jobs : int, default=2
        Number of CPU threads for TensorFlow. Lower values reduce CPU load,
        keeping the system responsive during training.
    objective : str, default="maximize"
        Either "minimize" or "maximize".
    modifiers : list of BaseModifier, optional
        List of modifiers to apply to function evaluations.

    Examples
    --------
    >>> from surfaces.test_functions.machine_learning import SimpleTransferLearningFunction
    >>> func = SimpleTransferLearningFunction(pretrain_epochs=2, finetune_epochs=3)
    >>> func.search_space
    {'freeze_ratio': [0.0, 0.5, 0.75, 1.0], 'learning_rate': [1e-05, 0.0001, 0.001], ...}
    >>> result = func({"freeze_ratio": 0.5, "learning_rate": 0.0001,
    ...                "dropout": 0.2, "dense_units": 64})
    >>> print(f"Validation accuracy: {result:.4f}")

    Notes
    -----
    Requires TensorFlow. Install with:
        pip install tensorflow

    Uses only native Keras datasets - no external model weights or large datasets
    are downloaded. MNIST and Fashion-MNIST are cached locally after first use.
    """

    name = "Simple Transfer Learning"
    _name_ = "simple_transfer_learning"
    __name__ = "SimpleTransferLearningFunction"

    para_names = ["freeze_ratio", "learning_rate", "dropout", "dense_units"]
    freeze_ratio_default = [0.0, 0.5, 0.75, 1.0]
    learning_rate_default = [1e-5, 1e-4, 1e-3]
    dropout_default = [0.0, 0.2, 0.4]
    dense_units_default = [32, 64, 128]

    def __init__(
        self,
        pretrain_epochs: int = 3,
        finetune_epochs: int = 5,
        batch_size: int = 64,
        pretrain_subset: int = 5000,
        finetune_subset: int = 5000,
        n_jobs: int = 2,
        objective: str = "maximize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
        use_surrogate: bool = False,
    ):
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.batch_size = batch_size
        self.pretrain_subset = pretrain_subset
        self.finetune_subset = finetune_subset
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
        """Search space for transfer learning optimization."""
        return {
            "freeze_ratio": self.freeze_ratio_default,
            "learning_rate": self.learning_rate_default,
            "dropout": self.dropout_default,
            "dense_units": self.dense_units_default,
        }

    def _create_objective_function(self) -> None:
        """Create objective function for transfer learning."""
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        # Limit CPU threads to keep system responsive
        tf.config.threading.set_inter_op_parallelism_threads(self.n_jobs)
        tf.config.threading.set_intra_op_parallelism_threads(self.n_jobs)

        # Load MNIST (source domain) for pretraining
        (X_mnist_train, y_mnist_train), _ = keras.datasets.mnist.load_data()
        X_mnist_train = X_mnist_train.astype("float32") / 255.0
        X_mnist_train = X_mnist_train[..., np.newaxis]  # Add channel dim

        # Load Fashion-MNIST (target domain) for fine-tuning
        (X_fashion_train, y_fashion_train), (X_fashion_val, y_fashion_val) = (
            keras.datasets.fashion_mnist.load_data()
        )
        X_fashion_train = X_fashion_train.astype("float32") / 255.0
        X_fashion_val = X_fashion_val.astype("float32") / 255.0
        X_fashion_train = X_fashion_train[..., np.newaxis]
        X_fashion_val = X_fashion_val[..., np.newaxis]

        # Subsample for faster evaluation
        rng = np.random.RandomState(42)

        if self.pretrain_subset < len(X_mnist_train):
            idx = rng.choice(len(X_mnist_train), self.pretrain_subset, replace=False)
            X_mnist_train = X_mnist_train[idx]
            y_mnist_train = y_mnist_train[idx]

        if self.finetune_subset < len(X_fashion_train):
            idx = rng.choice(len(X_fashion_train), self.finetune_subset, replace=False)
            X_fashion_train = X_fashion_train[idx]
            y_fashion_train = y_fashion_train[idx]

        # Smaller validation set
        val_size = min(2000, len(X_fashion_val))
        X_fashion_val = X_fashion_val[:val_size]
        y_fashion_val = y_fashion_val[:val_size]

        pretrain_epochs = self.pretrain_epochs
        finetune_epochs = self.finetune_epochs
        batch_size = self.batch_size

        def objective_function(params: Dict[str, Any]) -> float:
            # Build base CNN (feature extractor)
            base_input = layers.Input(shape=(28, 28, 1))
            x = layers.Conv2D(32, 3, activation="relu", padding="same", name="conv1")(base_input)
            x = layers.MaxPooling2D(2, name="pool1")(x)
            x = layers.Conv2D(64, 3, activation="relu", padding="same", name="conv2")(x)
            x = layers.MaxPooling2D(2, name="pool2")(x)
            x = layers.Flatten(name="flatten")(x)

            base_model = keras.Model(base_input, x, name="feature_extractor")

            # Build pretrain model (MNIST classifier)
            pretrain_output = layers.Dense(64, activation="relu", name="pretrain_dense")(
                base_model.output
            )
            pretrain_output = layers.Dense(10, activation="softmax", name="pretrain_output")(
                pretrain_output
            )
            pretrain_model = keras.Model(base_model.input, pretrain_output)

            # Pretrain on MNIST
            pretrain_model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            pretrain_model.fit(
                X_mnist_train,
                y_mnist_train,
                batch_size=batch_size,
                epochs=pretrain_epochs,
                verbose=0,
            )

            # Freeze layers based on freeze_ratio
            freeze_ratio = params["freeze_ratio"]
            # Freezable layers: conv1, pool1, conv2, pool2, flatten
            freezable_layers = ["conv1", "pool1", "conv2", "pool2", "flatten"]
            n_freeze = int(len(freezable_layers) * freeze_ratio)

            for layer_name in freezable_layers[:n_freeze]:
                base_model.get_layer(layer_name).trainable = False
            for layer_name in freezable_layers[n_freeze:]:
                base_model.get_layer(layer_name).trainable = True

            # Build fine-tune model (Fashion-MNIST classifier)
            finetune_input = base_model.input
            x = base_model.output
            x = layers.Dropout(params["dropout"])(x)
            x = layers.Dense(params["dense_units"], activation="relu")(x)
            finetune_output = layers.Dense(10, activation="softmax")(x)
            finetune_model = keras.Model(finetune_input, finetune_output)

            # Fine-tune on Fashion-MNIST
            finetune_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            history = finetune_model.fit(
                X_fashion_train,
                y_fashion_train,
                batch_size=batch_size,
                epochs=finetune_epochs,
                validation_data=(X_fashion_val, y_fashion_val),
                verbose=0,
            )

            # Return final validation accuracy
            val_accuracy = history.history["val_accuracy"][-1]
            return val_accuracy

        self.pure_objective_function = objective_function
