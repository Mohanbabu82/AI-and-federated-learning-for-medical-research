"""Shared plotting style: colorblind-safe palette (Okabe-Ito), readable fonts, consistent
per-method colors/markers (redundant encoding so lines are distinguishable even in grayscale).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

METHOD_STYLE = {
    "centralized": {"color": OKABE_ITO["black"], "marker": "s", "linestyle": "--", "label": "Centralized"},
    "local_only": {"color": OKABE_ITO["vermillion"], "marker": "^", "linestyle": ":", "label": "Local-only"},
    "fedavg": {"color": OKABE_ITO["sky_blue"], "marker": "o", "linestyle": "-", "label": "FedAvg"},
    "fedprox": {"color": OKABE_ITO["blue"], "marker": "D", "linestyle": "-.", "label": "FedProx"},
    "fedsaa": {"color": OKABE_ITO["bluish_green"], "marker": "*", "linestyle": "-", "label": "FedSAA (ours)"},
}

DPI = 300


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": DPI,
        "font.size": 12,
        "font.family": "sans-serif",
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        "figure.autolayout": True,
    })


def save_fig(fig, out_dir: str, name: str) -> list[str]:
    """Saves `fig` as both <name>.pdf and <name>.png (300 dpi) in out_dir. Returns the paths."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        paths.append(path)
    return paths
