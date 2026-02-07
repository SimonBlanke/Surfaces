"""Image Data Augmentation Optimization for Fashion-MNIST."""

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from surfaces.modifiers import BaseModifier

from ..._base_data_augmentation import BaseDataAugmentation


class ImageAugmentationFunction(BaseDataAugmentation):
    """Image Data Augmentation test function for Fashion-MNIST classification.

    **What is optimized:**
    This function optimizes data augmentation parameters to improve model
    robustness and generalization. Data augmentation artificially increases
    training data diversity by applying random transformations. The search includes:
    - Rotation range in degrees (0, 15, 30)
    - Width shift (0.0, 0.1, 0.2)
    - Height shift (0.0, 0.1, 0.2)
    - Horizontal flip (True/False)
    - Zoom range (0.0, 0.1, 0.2)

    **Data augmentation concept:**
    Data augmentation helps models learn invariances (e.g., objects rotated or
    shifted should still be recognized). Too little augmentation leads to overfitting,
    while too much can make training unstable or hurt performance by making
    examples unrealistic.

    **What the score means:**
    The score is the validation accuracy (0.0 to 1.0) achieved by a fixed CNN
    architecture trained with the specified augmentation pipeline on Fashion-MNIST.
    Higher scores indicate augmentation strategies that improve generalization
    without making the training task too difficult.

    **Optimization goal:**
    MAXIMIZE the validation accuracy. The goal is to find the optimal level of
    augmentation that:
    - Prevents overfitting by increasing training data diversity
    - Maintains realistic training examples (not too aggressive)
    - Helps the model learn useful invariances for the task

    **Computational cost:**
    Each evaluation trains a CNN with data augmentation, making this expensive.
    The default uses a subset of Fashion-MNIST and few epochs to keep evaluation time
    reasonable (~10-30 seconds per evaluation on CPU).

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
    >>> from surfaces.test_functions.machine_learning import ImageAugmentationFunction
    >>> func = ImageAugmentationFunction(n_epochs=5, subset_size=3000)
    >>> func.search_space
    {'rotation_range': [0, 15, 30], 'width_shift': [0.0, 0.1, 0.2], ...}
    >>> result = func({"rotation_range": 15, "width_shift": 0.1, "height_shift": 0.1,
    ...                "horizontal_flip": True, "zoom_range": 0.1})
    >>> print(f"Validation accuracy: {result:.4f}")

    Notes
    -----
    Requires TensorFlow. Install with:
        pip install tensorflow

    The function uses Fashion-MNIST (~29MB), a drop-in replacement for MNIST with
    clothing item classification. Dataset is cached locally after first use.
    """

    name = "Image Augmentation"
    _name_ = "image_augmentation"
    __name__ = "ImageAugmentationFunction"

    para_names = ["rotation_range", "width_shift", "height_shift", "horizontal_flip", "zoom_range"]
    rotation_range_default = [0, 15, 30]
    width_shift_default = [0.0, 0.1, 0.2]
    height_shift_default = [0.0, 0.1, 0.2]
    horizontal_flip_default = [True, False]
    zoom_range_default = [0.0, 0.1, 0.2]

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
        """Search space for image augmentation optimization."""
        return {
            "rotation_range": self.rotation_range_default,
            "width_shift": self.width_shift_default,
            "height_shift": self.height_shift_default,
            "horizontal_flip": self.horizontal_flip_default,
            "zoom_range": self.zoom_range_default,
        }

    def _create_objective_function(self) -> None:
        """Create objective function for image augmentation optimization."""
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
            # Create data augmentation pipeline
            datagen = ImageDataGenerator(
                rotation_range=params["rotation_range"],
                width_shift_range=params["width_shift"],
                height_shift_range=params["height_shift"],
                horizontal_flip=params["horizontal_flip"],
                zoom_range=params["zoom_range"],
            )

            # Build fixed CNN architecture
            model = keras.Sequential(
                [
                    layers.Conv2D(
                        32,
                        kernel_size=3,
                        activation="relu",
                        padding="same",
                        input_shape=(28, 28, 1),
                    ),
                    layers.MaxPooling2D(pool_size=(2, 2)),
                    layers.Conv2D(64, kernel_size=3, activation="relu", padding="same"),
                    layers.MaxPooling2D(pool_size=(2, 2)),
                    layers.Flatten(),
                    layers.Dense(128, activation="relu"),
                    layers.Dense(10, activation="softmax"),
                ]
            )

            # Compile model
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            # Train model with augmentation (suppress output)
            history = model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                epochs=n_epochs,
                validation_data=(X_val, y_val),
                steps_per_epoch=len(X_train) // batch_size,
                verbose=0,
            )

            # Return final validation accuracy
            val_accuracy = history.history["val_accuracy"][-1]
            return val_accuracy

        self.pure_objective_function = objective_function
