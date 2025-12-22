# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Regression datasets for ML test functions."""

from sklearn.datasets import load_diabetes, fetch_california_housing

# Pre-load datasets for fast access
diabetes_dataset = load_diabetes()
_california_dataset = None  # Lazy load (larger download)


def diabetes_data():
    """Load diabetes dataset (442 samples, 10 features)."""
    return diabetes_dataset.data, diabetes_dataset.target


def california_data():
    """Load California housing dataset (20640 samples, 8 features)."""
    global _california_dataset
    if _california_dataset is None:
        _california_dataset = fetch_california_housing()
    return _california_dataset.data, _california_dataset.target


# Registry for easy access
DATASETS = {
    "diabetes": diabetes_data,
    "california": california_data,
}
