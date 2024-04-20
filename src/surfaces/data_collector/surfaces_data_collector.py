from search_data_collector import SqlSearchData

from .config import default_search_data_path


class SurfacesDataCollector(SqlSearchData):
    def __init__(self, path=None) -> None:
        if path is None:
            path = default_search_data_path
        super().__init__(path, func2str=True)
