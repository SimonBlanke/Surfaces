# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
import pandas as pd

from .._base_test_function import BaseTestFunction
from ...data_collector import SurfacesDataCollector


class MachineLearningFunction(BaseTestFunction):
    def __init__(self, *args, sleep=0, evaluate_from_data=False, **kwargs):
        super().__init__(*args, sleep, **kwargs)

        if evaluate_from_data:
            self.sdc = SurfacesDataCollector()
            self._objective_function_ = self.objective_function_loaded
        else:
            self._objective_function_ = self.pure_objective_function

    def objective_function_loaded(self, params):
        try:
            parameter_d = params.para_dict
        except AttributeError:
            parameter_d = params

        for para_names, dim_value in parameter_d.items():
            try:
                parameter_d[para_names] = dim_value.__name__
            except AttributeError:
                pass

        search_data = self.sdc.load(self.__name__)
        if search_data is None:
            msg = "Search Data is empty"
            raise TypeError(msg)

        params_df = pd.DataFrame(parameter_d, index=[0])

        para_df_row = search_data[
            np.all(search_data[self.para_names].values == params_df.values, axis=1)
        ]
        score = para_df_row["score"].values[0]
        return score
