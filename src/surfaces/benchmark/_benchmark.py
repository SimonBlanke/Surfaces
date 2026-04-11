"""Central Benchmark class for gradient-free optimizer comparison.

The Benchmark object holds configuration (budget, seeds), a registry
of test functions and optimizer specs, and accumulated trace data.
Each call to run() executes only the missing (function, optimizer, seed)
combinations, making incremental benchmarking natural.
"""

from __future__ import annotations

import importlib
import json
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from surfaces._cost import calibrate

if TYPE_CHECKING:
    from surfaces.benchmark._accessors import IOAccessor, PlotAccessor, ResultAccessor
    from surfaces.benchmark._backends import ParallelBackend
from surfaces.benchmark._progress import TrialInfo, _ProgressBar
from surfaces.benchmark._resolve import resolve_functions, resolve_optimizer
from surfaces.benchmark._runner import (
    _instantiate_function,
    _run_ask_tell,
    _run_sealed,
    _run_trial,
    _TrialResult,
)
from surfaces.benchmark._suites import Suite
from surfaces.benchmark._trace import EvalRecord, Trace


class Benchmark:
    """Configurable, incremental benchmark runner.

    Collects test functions and optimizer specs via add methods,
    then runs only the missing combinations on each run() call.
    Results accumulate in the instance across runs.

    Parameters
    ----------
    budget_cu : float, optional
        Maximum compute budget per run in Compute Units.
    budget_iter : int, optional
        Maximum number of function evaluations per run.
    n_seeds : int
        Number of independent runs per (function, optimizer) pair.
    seed : int
        Base random seed. Run i uses seed + i.
    catch : str
        Error handling for individual trials. ``"raise"`` (default)
        propagates exceptions immediately. ``"warn"`` logs a warning
        and continues. ``"skip"`` silently skips the failed trial.
        Failed trials are recorded in ``bench.errors`` regardless of mode.

    Examples
    --------
    >>> bench = Benchmark(budget_cu=50_000, n_seeds=5)
    >>> bench.add_functions(collection.filter(category="bbob"))
    >>> bench.add_optimizers([HillClimbing, RandomSearch])
    >>> bench.run()
    >>> bench.results.summary()
    """

    _CATCH_MODES = frozenset({"raise", "warn", "skip"})

    def __init__(
        self,
        budget_cu: float | None = None,
        budget_iter: int | None = None,
        n_seeds: int = 1,
        seed: int = 0,
        catch: str = "raise",
    ):
        if budget_cu is None and budget_iter is None:
            raise ValueError("Specify at least one of budget_cu or budget_iter")
        if catch not in self._CATCH_MODES:
            raise ValueError(f"catch must be one of {sorted(self._CATCH_MODES)}, got {catch!r}")

        self._budget_cu = budget_cu
        self._budget_iter = budget_iter
        self._n_seeds = n_seeds
        self._seed = seed
        self._catch = catch

        self._functions: list[type] = []
        self._optimizers: list[tuple[Any, dict]] = []
        self._traces: dict[tuple[str, str, int], Trace] = {}
        self._errors: dict[tuple[str, str, int], Exception] = {}
        self._calibration_ref_time: float | None = None

        from surfaces.benchmark._accessors import IOAccessor, PlotAccessor, ResultAccessor

        self._results_accessor = ResultAccessor(self)
        self._io_accessor = IOAccessor(self)
        self._plot_accessor = PlotAccessor(self)

    def add_functions(self, functions: Any) -> Benchmark:
        """Add test functions to the benchmark.

        Accepts a single class, a list of classes, or a Collection.
        Duplicates are silently ignored.
        """
        resolved = resolve_functions(functions)
        for func_cls in resolved:
            if func_cls not in self._functions:
                self._functions.append(func_cls)
        return self

    def add_optimizers(self, optimizers: Any) -> Benchmark:
        """Add optimizer specs to the benchmark.

        Accepts a single spec or a list of specs. Each spec can be
        a class (auto-detected by module path) or a (class, params_dict)
        tuple. Duplicates are silently ignored.
        """
        if isinstance(optimizers, list):
            specs = optimizers
        else:
            specs = [optimizers]

        for spec in specs:
            normalized = self._normalize_optimizer_spec(spec)
            if normalized not in self._optimizers:
                self._optimizers.append(normalized)
        return self

    def remove_functions(self, functions: Any) -> Benchmark:
        """Remove test functions and their associated traces.

        Accepts a single class, a list of classes, or a Collection.
        Classes not currently registered are silently ignored.
        All traces recorded for removed functions are deleted.
        """
        resolved = resolve_functions(functions)
        for func_cls in resolved:
            if func_cls in self._functions:
                self._functions.remove(func_cls)
                func_name = func_cls.__name__
                self._purge_traces(function=func_name)
        return self

    def remove_optimizers(self, optimizers: Any) -> Benchmark:
        """Remove optimizer specs and their associated traces.

        Accepts a single spec or a list of specs. Each spec can be a
        bare class or a ``(class, params_dict)`` tuple.

        When a bare class is passed, **all** entries using that class
        are removed regardless of their params. When a tuple is passed,
        only the entry with exactly matching params is removed. This
        lets you selectively drop one configuration while keeping others::

            bench.remove_optimizers(TPESampler)                # all TPESampler entries
            bench.remove_optimizers((TPESampler, {"n": 10}))   # only this config

        Specs not currently registered are silently ignored.
        All traces recorded for removed optimizers are deleted.
        """
        if isinstance(optimizers, list):
            specs = optimizers
        else:
            specs = [optimizers]

        for spec in specs:
            if isinstance(spec, tuple):
                normalized = self._normalize_optimizer_spec(spec)
                if normalized in self._optimizers:
                    adapter = resolve_optimizer(self._to_spec(normalized))
                    self._optimizers.remove(normalized)
                    self._purge_traces(optimizer=adapter.name)
            else:
                to_remove = [(obj, p) for obj, p in self._optimizers if obj is spec]
                for entry in to_remove:
                    adapter = resolve_optimizer(self._to_spec(entry))
                    self._optimizers.remove(entry)
                    self._purge_traces(optimizer=adapter.name)
        return self

    def _purge_traces(
        self,
        function: str | None = None,
        optimizer: str | None = None,
    ) -> None:
        """Remove traces and errors matching the given function or optimizer name."""
        keys_to_drop = [
            k
            for k in self._traces
            if (function is not None and k[0] == function)
            or (optimizer is not None and k[1] == optimizer)
        ]
        for k in keys_to_drop:
            del self._traces[k]
            self._errors.pop(k, None)

    def run(
        self,
        *,
        verbose: bool = True,
        callback: Callable[[TrialInfo], None] | None = None,
        backend: ParallelBackend | None = None,
    ) -> Benchmark:
        """Run all missing (function, optimizer, seed) combinations.

        Only executes combinations that have no trace yet. New traces
        are added to the existing results, so previous data is preserved.

        Parameters
        ----------
        verbose : bool
            Print progress to stdout for each trial plus a summary line.
        callback : callable, optional
            Called with a ``TrialInfo`` after each trial (including skipped
            ones). Useful for custom progress bars or logging.
        backend : ParallelBackend, optional
            Parallel execution backend. When provided, pending trials are
            dispatched to workers via ``backend.map()``. When ``None``
            (the default), trials run sequentially in the current process.

            Note: with a parallel backend and ``catch != "raise"``, all
            trials complete before errors are collected. In sequential mode,
            ``catch="raise"`` stops on the first failure.

        Returns self for chaining: ``bench.run().results.summary()``
        """
        if not self._functions:
            raise ValueError("No functions added. Call add_functions() first.")
        if not self._optimizers:
            raise ValueError("No optimizers added. Call add_optimizers() first.")

        self._calibration_ref_time = calibrate()

        adapters = [resolve_optimizer(self._to_spec(opt)) for opt in self._optimizers]

        if backend is not None:
            self._run_parallel(adapters, backend, verbose, callback)
        else:
            self._run_sequential(adapters, verbose, callback)

        return self

    def _run_sequential(
        self,
        adapters: list,
        verbose: bool,
        callback: Callable[[TrialInfo], None] | None,
    ) -> None:
        """Execute trials one by one in the current process."""
        total = len(self._functions) * len(adapters) * self._n_seeds
        progress = (
            _ProgressBar(
                n_functions=len(self._functions),
                n_optimizers=len(adapters),
                n_seeds=self._n_seeds,
                total=total,
            )
            if verbose
            else None
        )
        trial_index = 0

        for func_cls in self._functions:
            func_name = func_cls.__name__

            for adapter in adapters:
                for i in range(self._n_seeds):
                    run_seed = self._seed + i
                    key = (func_name, adapter.name, run_seed)
                    trial_index += 1

                    if key in self._traces:
                        info = TrialInfo(
                            function=func_name,
                            optimizer=adapter.name,
                            seed=run_seed,
                            index=trial_index,
                            total=total,
                            skipped=True,
                            wall_seconds=None,
                        )
                        if progress:
                            progress.trial_complete(info)
                        if callback:
                            callback(info)
                        continue

                    try:
                        func = _instantiate_function(func_cls)

                        t0 = time.perf_counter()
                        if adapter.is_sealed:
                            trace = _run_sealed(
                                func, adapter, run_seed, self._budget_cu, self._budget_iter
                            )
                        else:
                            trace = _run_ask_tell(
                                func, adapter, run_seed, self._budget_cu, self._budget_iter
                            )
                        wall = time.perf_counter() - t0

                        self._traces[key] = trace

                    except Exception as exc:
                        if self._catch == "raise":
                            raise
                        self._errors[key] = exc
                        if self._catch == "warn":
                            warnings.warn(
                                f"Trial {func_name} x {adapter.name} "
                                f"(seed={run_seed}) failed: "
                                f"{type(exc).__name__}: {exc}",
                                stacklevel=2,
                            )
                        wall = None

                    info = TrialInfo(
                        function=func_name,
                        optimizer=adapter.name,
                        seed=run_seed,
                        index=trial_index,
                        total=total,
                        skipped=False,
                        wall_seconds=wall,
                        error=self._errors.get(key),
                    )
                    if progress:
                        progress.trial_complete(info)
                    if callback:
                        callback(info)

        if progress:
            progress.summary()

    def _run_parallel(
        self,
        adapters: list,
        backend: ParallelBackend,
        verbose: bool,
        callback: Callable[[TrialInfo], None] | None,
    ) -> None:
        """Dispatch pending trials to a parallel backend."""
        ref_time = self._calibration_ref_time

        adapter_to_spec = {
            adapter.name: self._to_spec(opt) for adapter, opt in zip(adapters, self._optimizers)
        }

        pending_tasks = []
        skipped_keys = []

        for func_cls in self._functions:
            func_name = func_cls.__name__
            for adapter in adapters:
                opt_spec = adapter_to_spec[adapter.name]
                for i in range(self._n_seeds):
                    run_seed = self._seed + i
                    key = (func_name, adapter.name, run_seed)

                    if key in self._traces:
                        skipped_keys.append(key)
                        continue

                    task = (
                        key,
                        func_cls,
                        opt_spec,
                        run_seed,
                        self._budget_cu,
                        self._budget_iter,
                        ref_time,
                    )
                    pending_tasks.append(task)

        total = len(pending_tasks) + len(skipped_keys)

        if verbose:
            n_workers = backend.effective_n_jobs
            print(
                f"Benchmark: dispatching {len(pending_tasks)} trials "
                f"to {n_workers} workers ({len(skipped_keys)} skipped)"
            )

        trial_index = 0

        for key in skipped_keys:
            trial_index += 1
            if callback:
                callback(
                    TrialInfo(
                        function=key[0],
                        optimizer=key[1],
                        seed=key[2],
                        index=trial_index,
                        total=total,
                        skipped=True,
                        wall_seconds=None,
                    )
                )

        if pending_tasks:
            t_start = time.perf_counter()
            results: list[_TrialResult] = backend.map(_run_trial, pending_tasks)
            t_total = time.perf_counter() - t_start

            first_error = None
            for result in results:
                trial_index += 1

                if result.error is not None:
                    self._errors[result.key] = result.error
                    if first_error is None:
                        first_error = result.error
                    if self._catch == "warn":
                        warnings.warn(
                            f"Trial {result.key[0]} x {result.key[1]} "
                            f"(seed={result.key[2]}) failed: "
                            f"{type(result.error).__name__}: {result.error}",
                            stacklevel=2,
                        )
                else:
                    self._traces[result.key] = result.trace

                if callback:
                    callback(
                        TrialInfo(
                            function=result.key[0],
                            optimizer=result.key[1],
                            seed=result.key[2],
                            index=trial_index,
                            total=total,
                            skipped=False,
                            wall_seconds=result.wall_seconds,
                            error=result.error,
                        )
                    )

            if verbose:
                n_new = sum(1 for r in results if r.error is None)
                n_failed = sum(1 for r in results if r.error is not None)
                from surfaces.benchmark._progress import _format_time

                print(
                    f"[done] {n_new} new, {len(skipped_keys)} skipped, "
                    f"{n_failed} failed, {_format_time(t_total)} total"
                )

            if first_error is not None and self._catch == "raise":
                raise first_error

        elif verbose:
            print("[done] all trials already cached")

    @property
    def results(self) -> ResultAccessor:
        """Access benchmark results: summary, traces, dataframe export."""
        return self._results_accessor

    @property
    def io(self) -> IOAccessor:
        """Access save/load functionality."""
        return self._io_accessor

    @property
    def plot(self) -> PlotAccessor:
        """Access benchmark visualizations."""
        return self._plot_accessor

    @property
    def errors(self) -> dict[tuple[str, str, int], Exception]:
        """Failed trials from the last run, keyed by (function, optimizer, seed)."""
        return dict(self._errors)

    @classmethod
    def load(cls, path: str | Path) -> Benchmark:
        """Load a benchmark (config + results) from a JSON file.

        Functions must originate from the surfaces package.
        A warning is emitted if the Surfaces version differs from
        the one used when saving.

        Parameters
        ----------
        path : str or Path
            Path to a JSON file previously created by ``io.save()``.
        """
        path = Path(path)

        with open(path) as f:
            data = json.load(f)

        version = data.get("format_version", 0)
        if version != 2:
            raise ValueError(
                f"Unsupported format version {version}. "
                f"This version of Surfaces supports format_version=2."
            )

        import surfaces

        saved_version = data.get("surfaces_version")
        if saved_version and saved_version != surfaces.__version__:
            warnings.warn(
                f"Benchmark was saved with Surfaces {saved_version}, "
                f"current version is {surfaces.__version__}. "
                f"Results may not be comparable.",
                stacklevel=2,
            )

        config = data["config"]
        bench = cls(
            budget_cu=config.get("budget_cu"),
            budget_iter=config.get("budget_iter"),
            n_seeds=config.get("n_seeds", 1),
            seed=config.get("seed", 0),
            catch=config.get("catch", "raise"),
        )

        for func_path in data.get("functions", []):
            if not func_path.startswith("surfaces."):
                raise ValueError(
                    f"Cannot load non-Surfaces function: {func_path}. "
                    f"Only functions from the surfaces package are supported."
                )
            module_path, class_name = func_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            func_cls = getattr(module, class_name)
            bench._functions.append(func_cls)

        for opt_entry in data.get("optimizers", []):
            class_path = opt_entry["class"]
            params = opt_entry.get("params", {})
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            opt_cls = getattr(module, class_name)
            bench._optimizers.append((opt_cls, params))

        for entry in data.get("traces", []):
            key = (entry["function"], entry["optimizer"], entry["seed"])
            trace = Trace()
            for rec in entry["records"]:
                trace.append(
                    EvalRecord(
                        params=rec["params"],
                        score=rec["score"],
                        eval_cu=rec["eval_cu"],
                        overhead_cu=rec["overhead_cu"],
                        cumulative_cu=rec["cumulative_cu"],
                        best_so_far=rec["best_so_far"],
                        wall_seconds=rec["wall_seconds"],
                    )
                )
            bench._traces[key] = trace

        return bench

    @classmethod
    def from_suite(cls, suite: Suite, **overrides: Any) -> Benchmark:
        """Create a Benchmark pre-configured from a Suite definition.

        The suite provides function filters and default budget/seed
        settings. Add optimizers and call run() to execute.

        Parameters
        ----------
        suite : Suite
            A predefined suite from ``surfaces.benchmark.suites``.
        **overrides
            Override suite defaults (budget_cu, budget_iter, n_seeds, seed).
        """
        from surfaces import collection

        kwargs: dict[str, Any] = {
            "budget_cu": suite.budget_cu,
            "budget_iter": suite.budget_iter,
            "n_seeds": suite.n_seeds,
        }
        kwargs.update(overrides)

        bench = cls(**kwargs)
        bench.add_functions(collection.filter(**suite.function_filter))
        return bench

    @staticmethod
    def _normalize_optimizer_spec(spec: Any) -> tuple[Any, dict]:
        """Normalize an optimizer spec to a (obj, params) tuple."""
        if isinstance(spec, tuple):
            if len(spec) != 2:
                raise TypeError(
                    f"Optimizer tuple must be (class, params_dict), got {len(spec)} elements"
                )
            obj, params = spec
            if not isinstance(params, dict):
                raise TypeError(
                    f"Second element of optimizer tuple must be a dict, got {type(params).__name__}"
                )
            return (obj, params)
        return (spec, {})

    @staticmethod
    def _to_spec(normalized: tuple[Any, dict]) -> Any:
        """Convert a normalized (obj, params) back to a resolver-compatible spec."""
        obj, params = normalized
        if params:
            return (obj, params)
        return obj

    def __repr__(self) -> str:
        n_funcs = len(self._functions)
        n_opts = len(self._optimizers)
        n_traces = len(self._traces)
        n_errors = len(self._errors)
        parts = [f"{n_funcs} functions", f"{n_opts} optimizers", f"{n_traces} traces"]
        if n_errors:
            parts.append(f"{n_errors} errors")
        if self._budget_cu is not None:
            parts.append(f"budget_cu={self._budget_cu}")
        if self._budget_iter is not None:
            parts.append(f"budget_iter={self._budget_iter}")
        return f"Benchmark({', '.join(parts)})"
