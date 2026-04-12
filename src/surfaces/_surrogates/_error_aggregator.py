# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Deduplicated error logging for long-running evaluation loops.

Instead of logging every single eval failure (which can produce thousands
of identical lines), the ErrorAggregator logs the first occurrence of each
distinct error type in full detail, suppresses duplicates, and emits
periodic summaries so the log file always reflects current state even
if the process crashes.
"""

from collections import Counter, OrderedDict


class ErrorAggregator:
    """Collect and deduplicate errors during data collection.

    Parameters
    ----------
    logger : logging.Logger, optional
        Logger instance. When None, errors are only counted.
    periodic_interval : int
        Emit a periodic count update to the log every N occurrences
        of the same error. Keeps the file log useful after a crash
        without producing thousands of identical lines.
    """

    def __init__(self, logger=None, periodic_interval: int = 500):
        self._logger = logger
        self._interval = periodic_interval
        self._counts: Counter = Counter()
        self._first_detail: OrderedDict = OrderedDict()

    def record(self, exc: Exception, context: str):
        """Record an error occurrence.

        Logs the first occurrence at DEBUG with full detail. Subsequent
        identical errors are suppressed, with periodic count updates
        every `periodic_interval` occurrences.
        """
        key = (type(exc).__name__, self._normalize_message(str(exc)))
        self._counts[key] += 1
        count = self._counts[key]

        if count == 1:
            self._first_detail[key] = context
            self._log(
                f"  Eval failed: {context} -> {key[0]}: {exc}",
                "debug",
            )
        elif count % self._interval == 0:
            self._log(
                f"  {key[0]}: {key[1]}... ({count}x so far)",
                "debug",
            )

    @property
    def total(self) -> int:
        return sum(self._counts.values())

    @property
    def distinct_count(self) -> int:
        return len(self._counts)

    def summary_lines(self) -> list[str]:
        """Return aggregated summary as a list of formatted strings."""
        lines = []
        for (etype, msg), count in self._counts.most_common():
            lines.append(f"    {etype}: {msg} ({count}x)")
        return lines

    def log_summary(self, prefix: str = "Error summary"):
        """Write the aggregated summary to the logger."""
        if not self._counts:
            return
        self._log(f"  {prefix} ({self.total} total, {self.distinct_count} distinct):", "info")
        for line in self.summary_lines():
            self._log(line, "info")

    def _normalize_message(self, msg: str) -> str:
        """Truncate to a stable prefix for grouping."""
        return msg[:100]

    def _log(self, msg: str, level: str = "info"):
        if self._logger:
            getattr(self._logger, level)(msg)
