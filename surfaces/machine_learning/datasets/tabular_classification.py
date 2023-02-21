# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from sklearn.datasets import load_digits


digits_dataset = load_digits()


def digits_data():
    return digits_dataset.data, digits_dataset.target
