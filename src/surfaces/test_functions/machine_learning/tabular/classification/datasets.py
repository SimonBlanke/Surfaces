# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Classification datasets for ML test functions."""

from sklearn.datasets import (
    fetch_covtype,
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
)

# Pre-load small datasets for fast access
digits_dataset = load_digits()
iris_dataset = load_iris()
wine_dataset = load_wine()
breast_cancer_dataset = load_breast_cancer()
_covtype_dataset = None  # Lazy load (larger download)


def digits_data():
    """Load digits dataset (1797 samples, 64 features, 10 classes)."""
    return digits_dataset.data, digits_dataset.target


def iris_data():
    """Load iris dataset (150 samples, 4 features, 3 classes)."""
    return iris_dataset.data, iris_dataset.target


def wine_data():
    """Load wine dataset (178 samples, 13 features, 3 classes)."""
    return wine_dataset.data, wine_dataset.target


def breast_cancer_data():
    """Load breast cancer dataset (569 samples, 30 features, 2 classes)."""
    return breast_cancer_dataset.data, breast_cancer_dataset.target


def covtype_data():
    """Load covertype dataset (581012 samples, 54 features, 7 classes).

    Note: This is a large dataset. First call triggers download.
    For faster training, a 10% subsample is returned.
    """
    global _covtype_dataset
    if _covtype_dataset is None:
        _covtype_dataset = fetch_covtype()
    # Return 10% subsample for reasonable training time
    n_samples = len(_covtype_dataset.target) // 10
    return _covtype_dataset.data[:n_samples], _covtype_dataset.target[:n_samples]


# Registry for easy access
DATASETS = {
    "digits": digits_data,
    "iris": iris_data,
    "wine": wine_data,
    "breast_cancer": breast_cancer_data,
    "covtype": covtype_data,
}
