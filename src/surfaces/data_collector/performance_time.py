# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import time
import numpy as np


def get_reference_time():
    c_time = time.perf_counter()

    m1 = np.ones([100, 100, 10, 100])
    m2 = np.ones([100, 100, 100, 10])
    for _ in range(10):
        np.matmul(m1, m2)

    d_time = time.perf_counter() - c_time
    return d_time
