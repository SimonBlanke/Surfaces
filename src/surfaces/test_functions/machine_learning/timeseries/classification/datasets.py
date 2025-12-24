# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Time-series classification datasets."""

import numpy as np

# Lazy-loaded datasets
_gunpoint_data = None
_ecg_data = None
_synthetic_ts_data = None


def gunpoint_data():
    """Synthetic GunPoint-like dataset (200 samples, 150 time points, 2 classes).

    Binary classification: distinguishing between two hand gesture patterns.
    Based on the UCR GunPoint benchmark structure.

    Returns
    -------
    tuple
        (X, y) where X is shape (n_samples, n_timepoints) and y is class labels.
    """
    global _gunpoint_data
    if _gunpoint_data is None:
        np.random.seed(42)
        n_samples_per_class = 100
        n_timepoints = 150

        # Class 0: smooth curve pattern
        t = np.linspace(0, 2 * np.pi, n_timepoints)
        class0 = np.array(
            [np.sin(t) + 0.3 * np.random.randn(n_timepoints) for _ in range(n_samples_per_class)]
        )

        # Class 1: peaked pattern
        class1 = np.array(
            [
                np.sin(t) + 0.5 * np.sin(3 * t) + 0.3 * np.random.randn(n_timepoints)
                for _ in range(n_samples_per_class)
            ]
        )

        X = np.vstack([class0, class1])
        y = np.array([0] * n_samples_per_class + [1] * n_samples_per_class)

        # Shuffle
        indices = np.random.permutation(len(y))
        _gunpoint_data = (X[indices], y[indices])

    return _gunpoint_data


def ecg_data():
    """Synthetic ECG-like dataset (300 samples, 100 time points, 3 classes).

    Multi-class classification of heartbeat patterns.
    Based on ECG benchmark structure with normal, arrhythmia, and other patterns.

    Returns
    -------
    tuple
        (X, y) where X is shape (n_samples, n_timepoints) and y is class labels.
    """
    global _ecg_data
    if _ecg_data is None:
        np.random.seed(42)
        n_samples_per_class = 100
        n_timepoints = 100

        t = np.linspace(0, 4 * np.pi, n_timepoints)

        # Class 0: Normal heartbeat (regular pattern)
        class0 = np.array(
            [
                np.exp(-((t - np.pi) ** 2) / 0.5)
                + np.exp(-((t - 3 * np.pi) ** 2) / 0.5)
                + 0.1 * np.random.randn(n_timepoints)
                for _ in range(n_samples_per_class)
            ]
        )

        # Class 1: Arrhythmia (irregular spacing)
        class1 = np.array(
            [
                np.exp(-((t - 0.8 * np.pi) ** 2) / 0.3)
                + np.exp(-((t - 2.5 * np.pi) ** 2) / 0.3)
                + 0.15 * np.random.randn(n_timepoints)
                for _ in range(n_samples_per_class)
            ]
        )

        # Class 2: Different morphology
        class2 = np.array(
            [
                0.5 * np.sin(t)
                + np.exp(-((t - 2 * np.pi) ** 2) / 1.0)
                + 0.1 * np.random.randn(n_timepoints)
                for _ in range(n_samples_per_class)
            ]
        )

        X = np.vstack([class0, class1, class2])
        y = np.array(
            [0] * n_samples_per_class + [1] * n_samples_per_class + [2] * n_samples_per_class
        )

        # Shuffle
        indices = np.random.permutation(len(y))
        _ecg_data = (X[indices], y[indices])

    return _ecg_data


def synthetic_ts_data():
    """Synthetic periodic patterns dataset (400 samples, 80 time points, 4 classes).

    Four distinct periodic patterns for baseline testing.

    Returns
    -------
    tuple
        (X, y) where X is shape (n_samples, n_timepoints) and y is class labels.
    """
    global _synthetic_ts_data
    if _synthetic_ts_data is None:
        np.random.seed(42)
        n_samples_per_class = 100
        n_timepoints = 80

        t = np.linspace(0, 2 * np.pi, n_timepoints)

        # Class 0: Sine wave
        class0 = np.array(
            [np.sin(t) + 0.2 * np.random.randn(n_timepoints) for _ in range(n_samples_per_class)]
        )

        # Class 1: Cosine wave
        class1 = np.array(
            [np.cos(t) + 0.2 * np.random.randn(n_timepoints) for _ in range(n_samples_per_class)]
        )

        # Class 2: Sawtooth-like
        class2 = np.array(
            [
                (t % np.pi) / np.pi + 0.2 * np.random.randn(n_timepoints)
                for _ in range(n_samples_per_class)
            ]
        )

        # Class 3: Square-like
        class3 = np.array(
            [
                np.sign(np.sin(t)) + 0.2 * np.random.randn(n_timepoints)
                for _ in range(n_samples_per_class)
            ]
        )

        X = np.vstack([class0, class1, class2, class3])
        y = np.array(
            [0] * n_samples_per_class
            + [1] * n_samples_per_class
            + [2] * n_samples_per_class
            + [3] * n_samples_per_class
        )

        # Shuffle
        indices = np.random.permutation(len(y))
        _synthetic_ts_data = (X[indices], y[indices])

    return _synthetic_ts_data


# Registry for easy access
DATASETS = {
    "gunpoint": gunpoint_data,
    "ecg": ecg_data,
    "synthetic_ts": synthetic_ts_data,
}
