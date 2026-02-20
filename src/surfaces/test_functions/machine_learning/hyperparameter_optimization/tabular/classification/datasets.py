# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Classification datasets for ML test functions."""

_digits_dataset = None
_iris_dataset = None
_wine_dataset = None
_breast_cancer_dataset = None
_covtype_dataset = None


def digits_data():
    """Load digits dataset (1797 samples, 64 features, 10 classes)."""
    global _digits_dataset
    if _digits_dataset is None:
        from sklearn.datasets import load_digits

        _digits_dataset = load_digits()
    return _digits_dataset.data, _digits_dataset.target


def iris_data():
    """Load iris dataset (150 samples, 4 features, 3 classes)."""
    global _iris_dataset
    if _iris_dataset is None:
        from sklearn.datasets import load_iris

        _iris_dataset = load_iris()
    return _iris_dataset.data, _iris_dataset.target


def wine_data():
    """Load wine dataset (178 samples, 13 features, 3 classes)."""
    global _wine_dataset
    if _wine_dataset is None:
        from sklearn.datasets import load_wine

        _wine_dataset = load_wine()
    return _wine_dataset.data, _wine_dataset.target


def breast_cancer_data():
    """Load breast cancer dataset (569 samples, 30 features, 2 classes)."""
    global _breast_cancer_dataset
    if _breast_cancer_dataset is None:
        from sklearn.datasets import load_breast_cancer

        _breast_cancer_dataset = load_breast_cancer()
    return _breast_cancer_dataset.data, _breast_cancer_dataset.target


def covtype_data():
    """Load covertype dataset (581012 samples, 54 features, 7 classes).

    Note: This is a large dataset. First call triggers download.
    For faster training, a 10% subsample is returned.
    """
    global _covtype_dataset
    if _covtype_dataset is None:
        from sklearn.datasets import fetch_covtype

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
