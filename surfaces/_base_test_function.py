# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


class BaseTestFunction:
    explanation = """ """

    dimensions = " "
    formula = r" "
    global_minimum = r" "

    def __init__(self, metric="score", input_type="dictionary", sleep=0):
        self.metric = metric
        self.input_type = input_type
        self.sleep = sleep

    def load_search_data(self):
        try:
            dataframe = self.sql_data.load(self.__name__)
        except:
            print("Path 2 database: ", self.sql_data.path)
        return dataframe
