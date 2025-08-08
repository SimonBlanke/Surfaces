# search_data_storage.py
import os
import pandas as pd
from pathlib import Path
from typing import Optional, Union, Dict, Any
import warnings
import json


class FileSearchData:
    """
    File-based storage for search data collected from expensive objective functions.

    This class provides methods to save and load search data (parameters and results)
    to/from files, avoiding the need to recompute expensive objective functions.
    Supports Parquet (preferred) and CSV formats.
    """

    def __init__(self, base_path: Union[str, Path], func2str: bool = True):
        """
        Initialize the file-based search data storage.

        Parameters
        ----------
        base_path : str or Path
            Base directory where search data files will be stored
        func2str : bool
            If True, convert function objects to their string representations
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.func2str = func2str
        self.metadata_file = self.base_path / "metadata.json"
        self._load_metadata()

    def _load_metadata(self):
        """Load or initialize metadata tracking stored tables."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"tables": {}, "version": "1.0"}
            self._save_metadata()

    def _save_metadata(self):
        """Save metadata to disk."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _get_file_path(self, table: str, format: str = "parquet") -> Path:
        """
        Generate file path for a given table name.

        Parameters
        ----------
        table : str
            Table/function name
        format : str
            File format ('parquet' or 'csv')

        Returns
        -------
        Path
            Full path to the data file
        """
        extension = ".parquet" if format == "parquet" else ".csv"
        return self.base_path / f"{table}{extension}"

    def save(self, table: str, data: pd.DataFrame, if_exists: str = "append"):
        """
        Save search data to file.

        Parameters
        ----------
        table : str
            Name of the table (typically the objective function name)
        data : pd.DataFrame
            DataFrame containing search data (parameters and results)
        if_exists : str
            How to handle existing data:
            - 'append': Add new data to existing file
            - 'replace': Overwrite existing file
            - 'fail': Raise error if file exists
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(data)}")

        if data.empty:
            warnings.warn(f"Attempting to save empty DataFrame for table '{table}'")
            return

        # Try Parquet first (preferred format)
        try:
            self._save_parquet(table, data, if_exists)
        except Exception as parquet_error:
            # Fall back to CSV if Parquet fails
            warnings.warn(
                f"Failed to save as Parquet: {parquet_error}. "
                "Falling back to CSV format."
            )
            self._save_csv(table, data, if_exists)

        # Update metadata
        self.metadata["tables"][table] = {
            "rows": len(data),
            "columns": list(data.columns),
            "last_updated": pd.Timestamp.now().isoformat(),
        }
        self._save_metadata()

    def _save_parquet(self, table: str, data: pd.DataFrame, if_exists: str):
        """Save data in Parquet format."""
        file_path = self._get_file_path(table, "parquet")

        if file_path.exists():
            if if_exists == "fail":
                raise ValueError(f"Table '{table}' already exists")
            elif if_exists == "append":
                existing_data = pd.read_parquet(file_path)
                data = pd.concat([existing_data, data], ignore_index=True)
            # if_exists == "replace", we just overwrite

        data.to_parquet(file_path, index=False, compression="snappy")

    def _save_csv(self, table: str, data: pd.DataFrame, if_exists: str):
        """Save data in CSV format as fallback."""
        file_path = self._get_file_path(table, "csv")

        if file_path.exists():
            if if_exists == "fail":
                raise ValueError(f"Table '{table}' already exists")
            elif if_exists == "append":
                existing_data = pd.read_csv(file_path)
                data = pd.concat([existing_data, data], ignore_index=True)

        data.to_csv(file_path, index=False)

    def load(self, table: str) -> Optional[pd.DataFrame]:
        """
        Load search data from file.

        Parameters
        ----------
        table : str
            Name of the table to load

        Returns
        -------
        pd.DataFrame or None
            Loaded search data, or None if table doesn't exist
        """
        # Try Parquet first
        parquet_path = self._get_file_path(table, "parquet")
        if parquet_path.exists():
            try:
                return pd.read_parquet(parquet_path)
            except Exception as e:
                warnings.warn(f"Failed to read Parquet file: {e}")

        # Try CSV as fallback
        csv_path = self._get_file_path(table, "csv")
        if csv_path.exists():
            return pd.read_csv(csv_path)

        return None

    def list_tables(self) -> list:
        """
        List all available tables.

        Returns
        -------
        list
            Names of all stored tables
        """
        return list(self.metadata["tables"].keys())

    def delete_table(self, table: str):
        """
        Delete a table and its data.

        Parameters
        ----------
        table : str
            Name of the table to delete
        """
        # Remove files
        for format in ["parquet", "csv"]:
            file_path = self._get_file_path(table, format)
            if file_path.exists():
                file_path.unlink()

        # Update metadata
        if table in self.metadata["tables"]:
            del self.metadata["tables"][table]
            self._save_metadata()

    def get_info(self, table: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a stored table.

        Parameters
        ----------
        table : str
            Name of the table

        Returns
        -------
        dict or None
            Table metadata including row count, columns, and last update time
        """
        return self.metadata["tables"].get(table)
