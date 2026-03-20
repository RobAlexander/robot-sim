"""Entity-count distribution plots for a batch job."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .job import RunRecord

# Known ranges (must stay in sync with constants.py)
_PEOPLE_MAX    = 10
_TREES_MAX     = 20
_HEDGEHOGS_MAX = 2


def _bar(ax, values: list[int], max_val: int, colour: str, title: str,
         xlabel: str) -> None:
    """Draw a bar chart showing the count for each integer value 0..max_val."""
    bins = list(range(max_val + 1))
    counts = [values.count(v) for v in bins]
    ax.bar(bins, counts, color=colour, edgecolor="white", linewidth=0.6)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Runs")
    ax.set_xticks(bins)
    ax.set_xlim(-0.5, max_val + 0.5)
    ax.yaxis.get_major_locator().set_params(integer=True)


def plot_entity_stats(
    runs: list["RunRecord"],
    output_path: str | Path | None = None,
) -> None:
    """Generate a 2×2 figure: three count histograms + one heatmap.

    If *output_path* is given the figure is saved there; otherwise it is shown
    interactively.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    people    = [r.counts.get("num_people",    0) for r in runs]
    trees     = [r.counts.get("num_trees",     0) for r in runs]
    hedgehogs = [r.counts.get("num_hedgehogs", 0) for r in runs]

    n = len(runs)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"Entity-count distributions  ({n} run{'s' if n != 1 else ''})",
                 fontsize=13, fontweight="bold")

    # -- Histograms ----------------------------------------------------------

    _bar(axes[0, 0], people,    _PEOPLE_MAX,    "#4a90d9",
         "People per run",    "Number of people")
    _bar(axes[0, 1], trees,     _TREES_MAX,     "#5ba85b",
         "Trees per run",     "Number of trees")
    _bar(axes[1, 0], hedgehogs, _HEDGEHOGS_MAX, "#c47c2b",
         "Hedgehogs per run", "Number of hedgehogs")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # -- Heatmap: people × trees --------------------------------------------

    ax = axes[1, 1]
    grid = np.zeros((_PEOPLE_MAX + 1, _TREES_MAX + 1), dtype=int)
    for p, t in zip(people, trees):
        grid[p, t] += 1

    im = ax.imshow(
        grid,
        aspect="auto",
        origin="lower",
        cmap="YlOrRd",
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="Runs")
    ax.set_title("People \u00d7 Trees heatmap", fontsize=11, fontweight="bold")
    ax.set_xlabel("Trees")
    ax.set_ylabel("People")
    ax.set_xticks(range(0, _TREES_MAX + 1, 2))
    ax.set_yticks(range(_PEOPLE_MAX + 1))

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150)
    else:
        plt.show()

    plt.close(fig)
