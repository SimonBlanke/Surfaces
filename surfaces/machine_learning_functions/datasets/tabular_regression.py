# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from sklearn.datasets import load_diabetes


diabetes_dataset = load_diabetes()


def diabetes_data():
    return diabetes_dataset.data, diabetes_dataset.target
