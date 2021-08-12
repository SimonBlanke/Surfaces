import numpy as np
from sklearn.datasets import load_iris

from bbox_functions.test_functions import StyblinskiTangFunction
from bbox_functions.visualize import matplotlib_surface


iris_dataset = load_iris()
X, y = iris_dataset.data, iris_dataset.target


styblinski_tang_function = StyblinskiTangFunction(n_dim=2, metric="loss")

step_ = 0.1
min_ = 5
max_ = 5
search_space = {
    "x0": np.arange(-min_, max_, step_),
    "x1": np.arange(-min_, max_, step_),
}

matplotlib_surface(styblinski_tang_function, search_space).show()
