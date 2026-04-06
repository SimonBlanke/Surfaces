"""Progress reporting for benchmark runs."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class TrialInfo:
    """Information about a single benchmark trial, passed to callbacks.

    Parameters
    ----------
    function : str
        Name of the test function class.
    optimizer : str
        Display name of the optimizer adapter.
    seed : int
        Random seed used for this trial.
    index : int
        1-based index of this trial within the run.
    total : int
        Total number of trials (including skipped) in the run.
    skipped : bool
        True if the trial was skipped because a trace already existed.
    wall_seconds : float or None
        Wall-clock time for the trial. None when skipped.
    """

    function: str
    optimizer: str
    seed: int
    index: int
    total: int
    skipped: bool
    wall_seconds: float | None


def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins:02d}m"


class _ProgressBar:
    """Hierarchical multi-stage progress display.

    Renders a tree-structured view with one bar per loop dimension
    (functions > optimizers > seeds), making the nesting visually
    obvious. Falls back to line-by-line output when stdout is not
    a TTY.
    """

    _N_LINES = 7
    _BAR_WIDTH = 20
    _FILLED = "━"
    _EMPTY = "░"

    _PREFIXES = [
        "├─ Function",
        "│  └─ Optimizer",
        "│     └─ Seed",
    ]
    _PREFIX_WIDTH = max(len(p) for p in _PREFIXES) + 2

    def __init__(
        self,
        n_functions: int,
        n_optimizers: int,
        n_seeds: int,
        total: int,
    ) -> None:
        self._n_functions = n_functions
        self._n_optimizers = n_optimizers
        self._n_seeds = n_seeds
        self._total = total
        self._t_start = time.perf_counter()
        self._new = 0
        self._skipped = 0
        self._is_tty = sys.stdout.isatty()

        if self._is_tty:
            sys.stdout.write("\n" * self._N_LINES)
            sys.stdout.flush()

    def trial_complete(self, info: TrialInfo) -> None:
        if info.skipped:
            self._skipped += 1
        else:
            self._new += 1

        if self._is_tty:
            self._render(info)
        else:
            self._print_line(info)

    def summary(self) -> None:
        elapsed = time.perf_counter() - self._t_start
        elapsed_str = _format_time(elapsed)

        if self._is_tty:
            full_bar = self._FILLED * self._BAR_WIDTH
            pw = self._PREFIX_WIDTH
            fw = len(str(self._n_functions))
            ow = len(str(self._n_optimizers))
            sw = len(str(self._n_seeds))

            lines = [
                f"Benchmark  {self._total}/{self._total}  {elapsed_str}",
                "│",
                f"{self._PREFIXES[0]:<{pw}}{full_bar}  "
                f"{self._n_functions:>{fw}}/{self._n_functions}",
                f"{self._PREFIXES[1]:<{pw}}{full_bar}  "
                f"{self._n_optimizers:>{ow}}/{self._n_optimizers}",
                f"{self._PREFIXES[2]:<{pw}}{full_bar}  " f"{self._n_seeds:>{sw}}/{self._n_seeds}",
                "│",
                f"│  {self._new} new, {self._skipped} skipped",
            ]
            self._write_lines(lines)
        else:
            print(f"[done] {self._new} new, {self._skipped} skipped, " f"{elapsed_str} total")

    def _bar(self, fraction: float) -> str:
        filled = int(self._BAR_WIDTH * fraction)
        return self._FILLED * filled + self._EMPTY * (self._BAR_WIDTH - filled)

    def _render(self, info: TrialInfo) -> None:
        idx0 = info.index - 1
        func_idx = idx0 // (self._n_optimizers * self._n_seeds) + 1
        remainder = idx0 % (self._n_optimizers * self._n_seeds)
        opt_idx = remainder // self._n_seeds + 1
        seed_idx = remainder % self._n_seeds + 1

        elapsed = time.perf_counter() - self._t_start
        completed = info.index
        if completed > 0:
            eta = elapsed / completed * (self._total - completed)
            eta_str = f"ETA {_format_time(eta)}"
        else:
            eta_str = "ETA ??s"

        pw = self._PREFIX_WIDTH
        fw = len(str(self._n_functions))
        ow = len(str(self._n_optimizers))
        sw = len(str(self._n_seeds))

        lines = [
            f"Benchmark  {completed}/{self._total}  {eta_str}",
            "│",
            f"{self._PREFIXES[0]:<{pw}}"
            f"{self._bar(func_idx / self._n_functions)}  "
            f"{func_idx:>{fw}}/{self._n_functions}  {info.function}",
            f"{self._PREFIXES[1]:<{pw}}"
            f"{self._bar(opt_idx / self._n_optimizers)}  "
            f"{opt_idx:>{ow}}/{self._n_optimizers}  {info.optimizer}",
            f"{self._PREFIXES[2]:<{pw}}"
            f"{self._bar(seed_idx / self._n_seeds)}  "
            f"{seed_idx:>{sw}}/{self._n_seeds}",
            "│",
            f"│  {self._new} new, {self._skipped} skipped",
        ]
        self._write_lines(lines)

    def _write_lines(self, lines: list[str]) -> None:
        sys.stdout.write(f"\033[{self._N_LINES}F" + "".join(f"\033[K{line}\n" for line in lines))
        sys.stdout.flush()

    def _print_line(self, info: TrialInfo) -> None:
        """Fallback for non-TTY output."""
        width = len(str(self._total))
        idx = f"{info.index:>{width}}"
        status = "-- skipped" if info.skipped else f"{info.wall_seconds:.2f}s"
        print(
            f"[{idx}/{self._total}] {info.function} x {info.optimizer} "
            f"(seed={info.seed}) {status}"
        )
