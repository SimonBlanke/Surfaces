from search_data_collector import SqlSearchData

from .config import paths


class SurfacesDataCollector(SqlSearchData):
    def __init__(self) -> None:
        path = paths["search-data path"]
        super().__init__(path, func2str=True)
