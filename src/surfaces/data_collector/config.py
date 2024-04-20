# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import os

here_path = os.path.dirname(os.path.realpath(__file__))
default_search_data_path = os.path.abspath(
    os.path.join(here_path, "..", "search_data.db")
)
