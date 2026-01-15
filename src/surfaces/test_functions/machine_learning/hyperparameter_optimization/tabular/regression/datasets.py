# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Regression datasets for ML test functions."""

from sklearn.datasets import (
    fetch_california_housing,
    load_diabetes,
    make_friedman1,
    make_friedman2,
    make_regression,
)

# Pre-load small datasets for fast access
diabetes_dataset = load_diabetes()
_california_dataset = None  # Lazy load (larger download)

# Generate synthetic datasets (deterministic)
_friedman1_data = None
_friedman2_data = None
_linear_data = None


def diabetes_data():
    """Load diabetes dataset (442 samples, 10 features)."""
    return diabetes_dataset.data, diabetes_dataset.target


def california_data():
    """Load California housing dataset (20640 samples, 8 features)."""
    global _california_dataset
    if _california_dataset is None:
        _california_dataset = fetch_california_housing()
    return _california_dataset.data, _california_dataset.target


def friedman1_data():
    """Generate Friedman #1 dataset (1000 samples, 10 features).

    Non-linear regression with interactions.
    """
    global _friedman1_data
    if _friedman1_data is None:
        _friedman1_data = make_friedman1(n_samples=1000, n_features=10, random_state=42)
    return _friedman1_data


def friedman2_data():
    """Generate Friedman #2 dataset (1000 samples, 4 features).

    Non-linear regression with different structure than Friedman #1.
    """
    global _friedman2_data
    if _friedman2_data is None:
        _friedman2_data = make_friedman2(n_samples=1000, random_state=42)
    return _friedman2_data


def linear_data():
    """Generate linear regression dataset (1000 samples, 20 features).

    Useful for testing on simple linear problems.
    """
    global _linear_data
    if _linear_data is None:
        _linear_data = make_regression(
            n_samples=1000, n_features=20, n_informative=10, noise=10.0, random_state=42
        )
    return _linear_data


# Registry for easy access
DATASETS = {
    "diabetes": diabetes_data,
    "california": california_data,
    "friedman1": friedman1_data,
    "friedman2": friedman2_data,
    "linear": linear_data,
}
