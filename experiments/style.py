"""Shared figure style for all ZNN-Bench paper figures (Elsevier print).

Rules (validated palette, see the project dataviz notes):
* Categorical colours are assigned in FIXED order (never cycled); colour
  follows the entity (e.g. a discretization keeps its colour in every figure:
  use ``color_for(key)`` with a stable registry of keys).  More than 8 series
  -> small multiples or fold into "other".
* Every series also gets a marker / linestyle (secondary encoding, because
  three slots are below 3:1 contrast on white and figures are printed).
* >= 2 series -> legend always present; never a dual y-axis.
* Sequential (magnitude, heatmaps): one-hue blue ramp; diverging: blue <-> red
  with a neutral grey midpoint.
* Thin marks: 1.1 pt lines, 4 pt markers; recessive grid and axes.
* Paper figures: drawn at TEXTWIDTH inside ``print_rc()`` (text >= 7.5 pt at print size).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
TAB_DIR = ROOT / "results" / "tables"

# categorical palette, light mode, validated (worst adjacent CVD dE 9.1, normal 19.6)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (5, 2, 1, 2))]
GREY = "#8a8985"
INK = "#0b0b0b"
INK2 = "#52514e"

SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
            "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
DIV_NEUTRAL = "#f0efec"
DIV_BLUE, DIV_RED = "#1c5cab", "#c03a39"

CMAP_SEQ = LinearSegmentedColormap.from_list("znn_seq", SEQ_BLUE)
CMAP_DIV = LinearSegmentedColormap.from_list("znn_div", [DIV_BLUE, DIV_NEUTRAL, DIV_RED])

# Elsevier column widths (inches)
SINGLE = 3.54     # 90 mm
ONEHALF = 5.51    # 140 mm
DOUBLE = 7.48     # 190 mm

# Paper layout (figure revision of 26 Oct 2026).  The manuscript and its supplement
# (elsarticle preprint, 12 pt) include every data figure at \textwidth = 390 pt =
# 5.40 in (137 mm).  Paper figures are drawn at that width, so they print at scale ~1,
# inside ``print_rc()``: no text below 7.5 pt (tick labels and legends 8 pt, axis labels
# and titles 8.5 pt); lines and markers keep the thin-mark rule above.
TEXTWIDTH = 390.0 / 72.27
PRINT_RC = {
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5, "figure.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.title_fontsize": 8,
    "axes.titlepad": 3.5, "axes.labelpad": 2.5, "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
    "legend.handlelength": 2.0, "legend.handletextpad": 0.5, "legend.columnspacing": 1.2,
    "legend.borderaxespad": 0.3, "legend.labelspacing": 0.35,
    "lines.markersize": 4.5,
}


def print_rc():
    """rc context of the paper figures (drawn at TEXTWIDTH, see PRINT_RC)."""
    return mpl.rc_context(PRINT_RC)

_KEY_SLOTS: dict = {}


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5, "axes.axisbelow": True,
        "lines.linewidth": 1.1, "lines.markersize": 4, "lines.markeredgewidth": 0.6,
        "legend.frameon": False, "legend.handlelength": 2.2,
        "axes.prop_cycle": mpl.cycler(color=CAT),
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def register_keys(keys):
    """Fix the colour/marker slot of each entity key (call once with the full,
    stable list so colours never depend on which subset a figure shows)."""
    for k in keys:
        if k not in _KEY_SLOTS:
            _KEY_SLOTS[k] = len(_KEY_SLOTS)


def slot(key):
    if key not in _KEY_SLOTS:
        register_keys([key])
    return _KEY_SLOTS[key]


def color_for(key):
    i = slot(key)
    return CAT[i % len(CAT)]


def marker_for(key):
    return MARKERS[slot(key) % len(MARKERS)]


def linestyle_for(key):
    i = slot(key)
    return LINESTYLES[(i // len(CAT) + i) % len(LINESTYLES)]


def style_for(key):
    """Keyword arguments for plot(): colour + marker + linestyle of an entity."""
    return {"color": color_for(key), "marker": marker_for(key), "linestyle": linestyle_for(key)}


def savefig(fig, name):
    """Save as vector PDF (for LaTeX) and 300-dpi PNG (for previews)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png")
    plt.close(fig)
    return FIG_DIR / f"{name}.pdf"


apply()
