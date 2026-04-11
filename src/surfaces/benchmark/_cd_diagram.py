"""Critical Difference diagram rendering (Demsar 2006).

Visualizes average ranks of multiple algorithms on a horizontal axis.
Algorithms that are not statistically distinguishable are connected
by thick bars ("cliques").

Requires matplotlib (``pip install surfaces[viz]``).
"""

from __future__ import annotations

import math
from typing import Any


def _get_pvalue(
    name_a: str,
    name_b: str,
    pvalues: dict[tuple[str, str], float],
) -> float | None:
    """Look up a pairwise p-value regardless of key order."""
    if (name_a, name_b) in pvalues:
        return pvalues[(name_a, name_b)]
    if (name_b, name_a) in pvalues:
        return pvalues[(name_b, name_a)]
    return None


def find_cliques(
    sorted_names: list[str],
    pvalues: dict[tuple[str, str], float],
    alpha: float,
) -> list[tuple[int, int]]:
    """Find maximal non-significant intervals in rank-sorted algorithms.

    An interval ``[i, j]`` means all algorithms from index *i* to *j*
    (in rank-sorted order) are pairwise non-significant at the given
    alpha. Only maximal intervals are returned.

    Parameters
    ----------
    sorted_names : list[str]
        Algorithm names sorted by average rank (best first).
    pvalues : dict
        Corrected pairwise p-values keyed by ``(name_a, name_b)``.
    alpha : float
        Significance threshold.

    Returns
    -------
    list of (start_idx, end_idx) tuples
        Indices into ``sorted_names``.
    """
    k = len(sorted_names)
    if k < 2:
        return []

    intervals: list[tuple[int, int]] = []
    for i in range(k - 1):
        j_max = i
        for j in range(i + 1, k):
            # Only need to check new pairs involving index j,
            # since [i..j-1] was validated in the previous iteration.
            extends = True
            for a in range(i, j):
                p = _get_pvalue(sorted_names[a], sorted_names[j], pvalues)
                if p is None or p <= alpha:
                    extends = False
                    break
            if extends:
                j_max = j
            else:
                # [i..j] contains [i..j-1] as a subset, so no larger
                # interval starting at i can work either
                break
        if j_max > i:
            intervals.append((i, j_max))

    maximal = [
        (ci, cj)
        for ci, cj in intervals
        if not any(oi <= ci and oj >= cj and (oi, oj) != (ci, cj) for oi, oj in intervals)
    ]
    return maximal


def render_cd_diagram(
    avg_ranks: dict[str, float],
    pvalues: dict[tuple[str, str], float],
    alpha: float = 0.05,
    title: str | None = None,
    width: float = 8.0,
) -> Any:
    """Render a Critical Difference diagram with matplotlib.

    Algorithms are placed on a horizontal rank axis. Better-ranked
    algorithms (lower rank number) appear on the left, worse on the
    right. Thick bars connect groups of algorithms that are not
    statistically distinguishable at the given alpha level.

    Parameters
    ----------
    avg_ranks : dict[str, float]
        Average rank per algorithm (from Friedman test).
    pvalues : dict
        Corrected pairwise p-values.
    alpha : float
        Significance threshold for clique detection.
    title : str, optional
        Figure title. Defaults to include the alpha value.
    width : float
        Figure width in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for CD diagrams. " "Install with: pip install surfaces[viz]"
        ) from None

    sorted_items = sorted(avg_ranks.items(), key=lambda x: x[1])
    names = [item[0] for item in sorted_items]
    ranks = [item[1] for item in sorted_items]
    k = len(names)

    if k < 2:
        raise ValueError("CD diagram requires at least 2 algorithms")

    cliques = find_cliques(names, pvalues, alpha)

    n_left = math.ceil(k / 2)
    n_right = k - n_left
    max_side = max(n_left, n_right)

    rank_lo, rank_hi = 1, k
    name_step = 0.38
    clique_step = 0.25
    axis_y = 0.0

    names_depth = max_side * name_step
    cliques_depth = max(len(cliques), 0) * clique_step
    height = max(1.2 + (names_depth + 0.3 + cliques_depth) * 0.45, 2.2)

    fig, ax = plt.subplots(figsize=(width, height))

    # Rank axis
    ax.plot(
        [rank_lo - 0.1, rank_hi + 0.1],
        [axis_y, axis_y],
        color="black",
        linewidth=1.3,
        solid_capstyle="butt",
    )
    for r in range(int(rank_lo), int(rank_hi) + 1):
        ax.plot([r, r], [axis_y - 0.04, axis_y + 0.04], color="black", linewidth=1)
        ax.text(
            r,
            axis_y + 0.14,
            str(r),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Left side: best-ranked algorithms (indices 0 .. n_left-1)
    for i in range(n_left):
        y = axis_y - (i + 1) * name_step
        rank = ranks[i]
        ax.text(
            rank_lo - 0.15,
            y,
            names[i],
            ha="right",
            va="center",
            fontsize=9,
            family="monospace",
        )
        ax.plot([rank_lo - 0.08, rank], [y, y], color="black", linewidth=0.7)
        ax.plot([rank, rank], [y, axis_y], color="black", linewidth=0.7)

    # Right side: worst-ranked algorithms, listed top-down from the worst
    for i in range(n_right):
        y = axis_y - (i + 1) * name_step
        idx = k - 1 - i
        rank = ranks[idx]
        ax.text(
            rank_hi + 0.15,
            y,
            names[idx],
            ha="left",
            va="center",
            fontsize=9,
            family="monospace",
        )
        ax.plot([rank, rank_hi + 0.08], [y, y], color="black", linewidth=0.7)
        ax.plot([rank, rank], [y, axis_y], color="black", linewidth=0.7)

    # Clique bars below the name lines.
    # Minimum visual width so bars remain visible when ranks are close.
    min_bar_width = 0.15 * (rank_hi - rank_lo) / max(k - 1, 1)
    bar_y_start = axis_y - max_side * name_step - 0.35
    for i, (start_idx, end_idx) in enumerate(cliques):
        y = bar_y_start - i * clique_step
        r_lo = ranks[start_idx]
        r_hi = ranks[end_idx]
        if r_hi - r_lo < min_bar_width:
            mid = (r_lo + r_hi) / 2
            r_lo = mid - min_bar_width / 2
            r_hi = mid + min_bar_width / 2
        ax.plot(
            [r_lo, r_hi],
            [y, y],
            color="black",
            linewidth=4,
            solid_capstyle="butt",
        )

    # Axis limits and cleanup
    longest_name = max(len(n) for n in names)
    x_margin = max(longest_name * 0.062, 1.2)
    ax.set_xlim(rank_lo - x_margin, rank_hi + x_margin)

    bottom = bar_y_start - max(len(cliques) - 1, 0) * clique_step - 0.4
    ax.set_ylim(bottom, axis_y + 0.55)
    ax.axis("off")

    if title is None:
        title = f"Critical Difference Diagram (alpha={alpha})"
    ax.set_title(title, fontsize=11, pad=12)

    fig.tight_layout()
    return fig
