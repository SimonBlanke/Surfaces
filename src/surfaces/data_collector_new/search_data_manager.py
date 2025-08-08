# file: surfaces/search_data_manager.py
import os
import sqlite3
from typing import Final


class SearchDataManager:
    """
    Centralised helper responsible for
      1. deciding *where* search-data are stored,
      2. creating the directory lazily,
      3. returning open sqlite3 connections.
    No other class touches the filesystem directly.
    """

    _SUBDIR: Final[str] = "search_data"  # project-internal folder
    _EXT: Final[str] = ".sqlite3"  # file-suffix for every function

    @classmethod
    def _data_root(cls) -> str:
        # Put DBs next to this source file → works inside editable installs & wheels
        return os.path.join(os.path.dirname(__file__), cls._SUBDIR)

    # ---------- public helpers -------------------------------------------------
    @classmethod
    def get_db_path(cls, func_name: str) -> str:
        """
        Resolve absolute filename for a function identified through its `_name_` attr.
        """
        root = cls._data_root()
        os.makedirs(root, exist_ok=True)
        return os.path.join(root, f"{func_name}{cls._EXT}")

    @classmethod
    def connect(cls, func_name: str) -> sqlite3.Connection:
        """
        Return a connection with sensible default pragmas for read-heavy workloads.
        """
        conn = sqlite3.connect(cls.get_db_path(func_name))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn
