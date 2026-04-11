"""Parallel execution backends for benchmark runs.

Backends control how benchmark trials are distributed across workers.
The interface is intentionally minimal: implement ``map`` to execute
a callable over a list of tasks and return results.

Built-in backends use ``concurrent.futures`` from the standard library.
Custom backends (Dask, Ray, etc.) can subclass ``ParallelBackend``
and implement the same interface.

Example
-------
>>> bench.run(backend=ProcessBackend(n_jobs=4))
>>> bench.run(backend=ThreadBackend(n_jobs=8))
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable


class ParallelBackend(ABC):
    """Base class for parallel benchmark execution.

    Subclasses must implement :meth:`map` to distribute work across
    workers. The ``n_jobs`` attribute controls the number of workers.

    Parameters
    ----------
    n_jobs : int
        Number of parallel workers. Use ``-1`` for all available CPU cores.
    """

    def __init__(self, n_jobs: int = -1):
        if not isinstance(n_jobs, int) or (n_jobs < 1 and n_jobs != -1):
            raise ValueError(f"n_jobs must be a positive integer or -1, got {n_jobs!r}")
        self._n_jobs = n_jobs

    @property
    def n_jobs(self) -> int:
        """Number of workers requested (``-1`` means all cores)."""
        return self._n_jobs

    @property
    def effective_n_jobs(self) -> int:
        """Resolved worker count (always a positive integer)."""
        if self._n_jobs == -1:
            return os.cpu_count() or 1
        return self._n_jobs

    @abstractmethod
    def map(self, fn: Callable, tasks: list) -> list:
        """Execute ``fn(task)`` for each task and return results.

        Results must be returned in the same order as *tasks*.

        Parameters
        ----------
        fn : callable
            A picklable callable accepting a single positional argument.
        tasks : list
            Task arguments to distribute across workers.

        Returns
        -------
        list
            One result per task, in submission order.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_jobs={self._n_jobs})"


class ProcessBackend(ParallelBackend):
    """Process-based parallelism via ``ProcessPoolExecutor``.

    Each worker runs in a separate OS process, bypassing the GIL.
    Task arguments and return values must be picklable.

    Parameters
    ----------
    n_jobs : int
        Number of worker processes. Defaults to ``-1`` (all CPU cores).
    """

    def map(self, fn: Callable, tasks: list) -> list:
        with ProcessPoolExecutor(max_workers=self.effective_n_jobs) as pool:
            return list(pool.map(fn, tasks))


class ThreadBackend(ParallelBackend):
    """Thread-based parallelism via ``ThreadPoolExecutor``.

    Useful when the objective function or optimizer releases the GIL
    (e.g. calls into C extensions). Avoids the pickling overhead of
    process-based backends.

    Parameters
    ----------
    n_jobs : int
        Number of worker threads. Defaults to ``-1`` (all CPU cores).
    """

    def map(self, fn: Callable, tasks: list) -> list:
        with ThreadPoolExecutor(max_workers=self.effective_n_jobs) as pool:
            return list(pool.map(fn, tasks))
