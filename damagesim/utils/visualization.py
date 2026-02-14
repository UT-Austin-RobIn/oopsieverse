"""
Live health-bar visualization helpers.

These are used by both the OmniGibson and Robosuite environments for
real-time health display during data collection or debugging.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("TkAgg")  # Non-blocking backend
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def setup_live_health_bars(
    object_names: List[str],
    figsize: Tuple[int, int] = (8, 4),
) -> Tuple["Figure", "Axes", Dict[str, "matplotlib.patches.Rectangle"]]:
    """
    Create a matplotlib figure with horizontal health bars for each object.

    Returns:
        (fig, ax, bars_dict)
    """
    if not HAS_MPL:
        raise RuntimeError("matplotlib is required for health visualization")

    n = len(object_names)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xlabel("Health (%)")
    ax.set_title("Object Health")
    ax.set_yticks(range(n))
    ax.set_yticklabels(object_names)
    ax.invert_yaxis()

    bars_dict: Dict[str, matplotlib.patches.Rectangle] = {}
    for i, name in enumerate(object_names):
        bar = ax.barh(i, 100, color="green", edgecolor="black", height=0.6)[0]
        bars_dict[name] = bar

    plt.tight_layout()
    plt.ion()
    plt.show(block=False)
    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig, ax, bars_dict


def _health_color(health: float) -> str:
    if health > 75:
        return "green"
    if health > 50:
        return "gold"
    if health > 25:
        return "orange"
    return "red"


def update_live_health_bars(
    fig: "Figure",
    ax: "Axes",
    bars_dict: Dict[str, "matplotlib.patches.Rectangle"],
    current_health: Dict[str, float],
    object_names: List[str],
) -> bool:
    """
    Update the live health bars with new values.

    Returns:
        True if the figure is still open, False if it was closed.
    """
    if not HAS_MPL:
        return False

    if not plt.fignum_exists(fig.number):
        return False

    for name in object_names:
        h = current_health.get(name, 100.0)
        bar = bars_dict.get(name)
        if bar is not None:
            bar.set_width(max(0, h))
            bar.set_color(_health_color(h))

    try:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    except Exception:
        return False

    return True

