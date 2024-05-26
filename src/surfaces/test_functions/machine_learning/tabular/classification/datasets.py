# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from sklearn.datasets import load_digits, load_wine, load_iris


digits_dataset = load_digits()
wine_dataset = load_wine()
iris_dataset = load_iris()


def digits_data():
    return digits_dataset.data, digits_dataset.target


def wine_data():
    return wine_dataset.data, wine_dataset.target


def iris_data():
    return iris_dataset.data, iris_dataset.target
