from .config import paths

from search_data_collector import SqlDataCollector


class SurfacesDataCollector(SqlDataCollector):
    def __init__(self) -> None:
        path = paths["search-data path"]
        super().__init__(path, func2str=True)
