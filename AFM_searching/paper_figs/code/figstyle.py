"""Shared figure styling.

Colour is assigned by the *job* it does, not by taste:

* **categorical** -- the three gradient estimators (ED / AE / CAE) and the two
  schedules (smooth / stiff).  Hues are taken in fixed slot order from a
  pre-validated palette (slots 1-3 clear the all-pairs colour-vision-deficiency
  and normal-vision separation floors), and a given estimator keeps its colour in
  every figure of the report.
* **sequential** -- anything that is an ordered magnitude (evolution time T,
  system size N) uses one blue ramp, light = small, dark = large.  These are not
  categorical series and must not be given arbitrary hues.
* **diverging** -- the staggered magnetisation ``<AM>`` runs from -1 to +1 through
  a meaningful zero, so it uses blue <-> red with a neutral grey midpoint.
  ``<AM^2>`` runs from 0 to 1 with no meaningful midpoint, so it uses the plain
  sequential blue ramp instead.

Every figure keeps a legend whenever more than one series is drawn, so identity
never rests on colour alone.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- categorical slots (fixed order, never cycled) ------------------------- #
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
        "#008300", "#4a3aa7", "#e34948"]

METHOD_COLOR = {"ED": SLOT[0], "AE": SLOT[1], "CAE": SLOT[2]}
METHOD_LABEL = {"ED": "ED (exact GS)", "AE": "AE (uncached)", "CAE": "CAE (cached)"}
METHOD_STYLE = {"ED": "-", "AE": "-", "CAE": "--"}

SCHEDULE_COLOR = {"smooth": SLOT[0], "stiff": SLOT[1]}
SCHEDULE_LABEL = {"smooth": "smooth ramp", "stiff": "stiff ramp"}

# --- sequential blue ramp (steps 100 -> 700) ------------------------------- #
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]
# second sequential ramp, for a figure that has to show two magnitude scales
# at once (the palette's rule: the second takes the next categorical hue)
ORANGE_RAMP = ["#fbd9c9", "#f7bda2", "#f3a17c", "#ef8557", "#eb6834",
               "#d4552a", "#b04521", "#8b3619", "#682711"]

# --- chrome ---------------------------------------------------------------- #
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"
NEUTRAL_MID = "#f0efec"

CMAP_DIVERGING = LinearSegmentedColormap.from_list(
    "afm_div", ["#0d366b", "#2a78d6", "#9ec5f4", NEUTRAL_MID,
                "#f3a08e", "#e34948", "#8c1f1f"])
CMAP_SEQUENTIAL = LinearSegmentedColormap.from_list("afm_seq", BLUE_RAMP)


def ramp_colors(n, ramp=None, lo=0.28, hi=1.0):
    """``n`` evenly spaced colours from a sequential ramp (light -> dark).

    ``lo`` starts at ramp step 250 rather than 100: these are *ordinal* series
    (one line per T, per N), so even the lightest must stay legible against the
    chart surface, unlike a continuous heat map where near-zero may recede.
    """
    import numpy as np
    ramp = BLUE_RAMP if ramp is None else ramp
    cmap = LinearSegmentedColormap.from_list("tmp", ramp)
    return [cmap(x) for x in np.linspace(lo, hi, n)]


def use_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "axes.labelcolor": INK,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.color": INK,
        "lines.linewidth": 1.4,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "legend.labelcolor": INK2,
        "figure.constrained_layout.use": True,
    })


def finish(ax, xlabel=None, ylabel=None, title=None, legend=True, ncol=1):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, color=INK, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(ncol=ncol)
    return ax
