# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Storage backends for CustomTestFunction persistence.

This module provides storage backends for persisting evaluation data,
checkpoints, and experiment metadata.

Built-in Backends
-----------------
- InMemoryStorage: Default, no persistence (data lost on exit)
- SQLiteStorage: File-based persistence using Python's sqlite3

Custom Backends
---------------
Users can create custom storage backends by implementing the Storage protocol.
See the Storage class docstring for the required interface.

Example: Custom Redis Storage
-----------------------------
>>> from surfaces.custom_test_function.storage import Storage
>>> import redis
>>>
>>> class RedisStorage(Storage):
...     def __init__(self, host: str, experiment: str):
...         self.client = redis.Redis(host=host)
...         self._experiment = experiment
...
...     @property
...     def experiment(self) -> str:
...         return self._experiment
...
...     def save_evaluation(self, evaluation: dict) -> None:
...         key = f"{self._experiment}:evals"
...         self.client.rpush(key, json.dumps(evaluation))
...
...     def load_evaluations(self) -> list:
...         key = f"{self._experiment}:evals"
...         return [json.loads(e) for e in self.client.lrange(key, 0, -1)]
...
...     # ... implement other required methods
>>>
>>> # Use with CustomTestFunction
>>> func = CustomTestFunction(
...     objective_fn=my_func,
...     search_space={...},
...     storage=RedisStorage("localhost", "my-experiment"),
... )

Example: Custom PostgreSQL Storage
----------------------------------
>>> class PostgresStorage(Storage):
...     def __init__(self, connection_string: str, experiment: str):
...         import psycopg2
...         self.conn = psycopg2.connect(connection_string)
...         self._experiment = experiment
...         self._setup_tables()
...     # ... implement required methods
"""

from ._memory import InMemoryStorage
from ._protocol import Storage
from ._sqlite import SQLiteStorage

__all__ = [
    "Storage",
    "InMemoryStorage",
    "SQLiteStorage",
]
