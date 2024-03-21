import os

here_path = os.path.dirname(os.path.realpath(__file__))


paths = {
    "search-data path": os.path.abspath(
        os.path.join(here_path, "..", "search_data.db")
    ),
}
