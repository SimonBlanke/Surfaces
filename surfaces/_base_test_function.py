# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time

from .data_collector import SurfacesDataCollector


class BaseTestFunction:
    explanation = """ """

    dimensions = " "
    formula = r" "
    global_minimum = r" "

    def __init__(self):
        self.sql_data = SurfacesDataCollector()

    def load_search_data(self):
        return self.sql_data.load(self.__name__)

    def __call__(self, *input):
        time.sleep(self.sleep)

        if self.input_type == "dictionary":
            return self.objective_function_dict(*input)
        elif self.input_type == "arrays":
            return self.objective_function_np(*input)
