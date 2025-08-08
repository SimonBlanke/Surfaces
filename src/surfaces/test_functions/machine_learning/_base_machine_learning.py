# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
import pandas as pd

from .._base_test_function import BaseTestFunction
from ...data_collector import SurfacesDataCollector


class MachineLearningFunction(BaseTestFunction):
    def __init__(self, metric="loss", sleep=0, evaluate_from_data=False, collect_search_data=False, search_space=None, **kwargs):
        # Filter out ML-specific kwargs before passing to parent
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['collect_search_data', 'search_space']}
        super().__init__(metric, sleep)
        
        self.collect_search_data = collect_search_data
        self.search_space_data = search_space

        if evaluate_from_data:
            self.sdc = SurfacesDataCollector()
            self._objective_function_ = self.objective_function_loaded
        else:
            self._objective_function_ = self.pure_objective_function
    
    def evaluate(self, params):
        """Evaluate the function with given parameters."""
        if isinstance(params, (list, tuple)):
            # Convert list/tuple to dict using param_names if available
            if hasattr(self, 'param_names'):
                param_dict = {name: val for name, val in zip(self.param_names, params)}
            else:
                # Fallback to generic naming
                param_dict = {f'x{i}': val for i, val in enumerate(params)}
        else:
            param_dict = params
        
        return self._objective_function_(param_dict)

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
