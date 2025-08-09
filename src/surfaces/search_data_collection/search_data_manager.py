# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import sqlite3
import time
from typing import Dict, List, Any, Optional, Tuple


class SearchDataManager:
    """
    Manages search data storage and retrieval using SQLite database.
    
    This class provides centralized management of search data for machine learning
    test functions, storing parameter combinations and their evaluation results
    along with timing information for benchmarking purposes.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the SearchDataManager.
        
        Args:
            data_dir: Directory to store search data. If None, uses default location
                     in src/surfaces/search_data/
        """
        if data_dir is None:
            # Store in central location in project structure
            self.data_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "search_data"
            )
        else:
            self.data_dir = data_dir
            
        # Create directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
    
    def get_db_path(self, function_name: str) -> str:
        """
        Get the database file path for a given function name.
        
        Args:
            function_name: Name of the test function (from _name_ attribute)
            
        Returns:
            Absolute path to the SQLite database file
        """
        return os.path.join(self.data_dir, f"{function_name}.db")
    
    def create_table(self, function_name: str, parameter_names: List[str]) -> None:
        """
        Create a table for storing search data if it doesn't exist.
        
        Args:
            function_name: Name of the test function
            parameter_names: List of parameter names for the function
        """
        db_path = self.get_db_path(function_name)
        
        with sqlite3.connect(db_path) as conn:
            # Create parameter columns (all as TEXT to handle different types)
            param_columns = ", ".join([f"{name} TEXT" for name in parameter_names])
            
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS search_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {param_columns},
                score REAL,
                evaluation_time REAL,
                timestamp REAL
            )
            """
            
            conn.execute(create_table_sql)
            
            # Create index for faster parameter lookups
            param_index = ", ".join(parameter_names)
            create_index_sql = f"""
            CREATE INDEX IF NOT EXISTS idx_params ON search_data ({param_index})
            """
            conn.execute(create_index_sql)
            
            conn.commit()
    
    def store_evaluation(self, function_name: str, parameters: Dict[str, Any], 
                        score: float, evaluation_time: float) -> None:
        """
        Store a single evaluation result in the database.
        
        Args:
            function_name: Name of the test function
            parameters: Dictionary of parameter values
            score: Evaluation score/result
            evaluation_time: Time taken for evaluation in seconds
        """
        db_path = self.get_db_path(function_name)
        
        with sqlite3.connect(db_path) as conn:
            # Convert parameter values to strings for storage
            param_values = []
            param_names = []
            
            for key, value in parameters.items():
                param_names.append(key)
                if hasattr(value, '__name__'):  # Handle function objects
                    param_values.append(value.__name__)
                else:
                    param_values.append(str(value))
            
            # Insert data
            columns = param_names + ["score", "evaluation_time", "timestamp"]
            values = param_values + [score, evaluation_time, time.time()]
            
            placeholders = ", ".join(["?"] * len(values))
            columns_str = ", ".join(columns)
            
            insert_sql = f"""
            INSERT INTO search_data ({columns_str})
            VALUES ({placeholders})
            """
            
            conn.execute(insert_sql, values)
            conn.commit()
    
    def store_batch(self, function_name: str, parameter_names: List[str],
                   evaluations: List[Tuple[Dict[str, Any], float, float]]) -> None:
        """
        Store multiple evaluation results in batch for better performance.
        
        Args:
            function_name: Name of the test function
            parameter_names: List of parameter names
            evaluations: List of tuples (parameters, score, evaluation_time)
        """
        if not evaluations:
            return
            
        db_path = self.get_db_path(function_name)
        
        with sqlite3.connect(db_path) as conn:
            # Prepare batch insert
            columns = parameter_names + ["score", "evaluation_time", "timestamp"]
            columns_str = ", ".join(columns)
            placeholders = ", ".join(["?"] * len(columns))
            
            insert_sql = f"""
            INSERT INTO search_data ({columns_str})
            VALUES ({placeholders})
            """
            
            batch_data = []
            current_time = time.time()
            
            for parameters, score, eval_time in evaluations:
                # Convert parameter values to strings
                param_values = []
                for param_name in parameter_names:
                    value = parameters[param_name]
                    if hasattr(value, '__name__'):
                        param_values.append(value.__name__)
                    else:
                        param_values.append(str(value))
                
                row_data = param_values + [score, eval_time, current_time]
                batch_data.append(row_data)
            
            conn.executemany(insert_sql, batch_data)
            conn.commit()
    
    def lookup_evaluation(self, function_name: str, parameters: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """
        Look up a stored evaluation result.
        
        Args:
            function_name: Name of the test function
            parameters: Dictionary of parameter values to look up
            
        Returns:
            Tuple of (score, evaluation_time) if found, None otherwise
        """
        db_path = self.get_db_path(function_name)
        
        if not os.path.exists(db_path):
            return None
            
        with sqlite3.connect(db_path) as conn:
            # Build WHERE clause for parameter matching
            where_conditions = []
            where_values = []
            
            for key, value in parameters.items():
                where_conditions.append(f"{key} = ?")
                if hasattr(value, '__name__'):
                    where_values.append(value.__name__)
                else:
                    where_values.append(str(value))
            
            where_clause = " AND ".join(where_conditions)
            
            select_sql = f"""
            SELECT score, evaluation_time
            FROM search_data
            WHERE {where_clause}
            LIMIT 1
            """
            
            cursor = conn.execute(select_sql, where_values)
            result = cursor.fetchone()
            
            if result:
                return (result[0], result[1])
            return None
    
    def get_all_evaluations(self, function_name: str) -> List[Dict[str, Any]]:
        """
        Get all stored evaluations for a function.
        
        Args:
            function_name: Name of the test function
            
        Returns:
            List of dictionaries containing all stored data
        """
        db_path = self.get_db_path(function_name)
        
        if not os.path.exists(db_path):
            return []
            
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row  # Enable column access by name
            
            cursor = conn.execute("SELECT * FROM search_data")
            results = []
            
            for row in cursor.fetchall():
                row_dict = dict(row)
                results.append(row_dict)
            
            return results
    
    def clear_data(self, function_name: str) -> None:
        """
        Clear all stored data for a function.
        
        Args:
            function_name: Name of the test function
        """
        db_path = self.get_db_path(function_name)
        
        if os.path.exists(db_path):
            with sqlite3.connect(db_path) as conn:
                conn.execute("DELETE FROM search_data")
                conn.commit()
    
    def get_database_info(self, function_name: str) -> Dict[str, Any]:
        """
        Get information about the database (size, record count, etc.).
        
        Args:
            function_name: Name of the test function
            
        Returns:
            Dictionary containing database statistics
        """
        db_path = self.get_db_path(function_name)
        
        if not os.path.exists(db_path):
            return {"exists": False}
            
        info = {"exists": True, "path": db_path}
        
        try:
            info["size_bytes"] = os.path.getsize(db_path)
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM search_data")
                info["record_count"] = cursor.fetchone()[0]
                
                # Get column information
                cursor = conn.execute("PRAGMA table_info(search_data)")
                columns = cursor.fetchall()
                info["columns"] = [col[1] for col in columns]  # Column names
                
        except Exception as e:
            info["error"] = str(e)
            
        return info