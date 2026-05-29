"""
Shared matplotlib style for the prediction market microstructure paper.
Provides a consistent academic visual language across all checkpoint figures.
"""
from __future__ import annotations

import contextlib
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# ─── Colour palette ───────────────────────────────────────────────────────────
BLUE    = "#4361EE"   # maker, primary series, trends
GREEN   = "#06D6A0"   # positive returns, YES, gains
RED     = "#EF233C"   # taker, negative returns, NO, losses
AMBER   = "#F9C74F"   # reference lines, spread, neutral zone
PURPLE  = "#8338EC"   # accent, third series
ORANGE  = "#FB5607"   # secondary accent
GRAY    = "#6C757D"   # muted gridlines, zero axes, minor labels
DARK    = "#212529"   # text colour
GRID_C  = "#DEE2E6"   # grid line colour
LIGHT   = "#F8F9FA"   # subtle panel tint

# Calibration-curve diverging colours (negative delta = underestimated → red)
CMAP_DIV = "RdYlGn"   # use for δ_b scatter colouring

# Category group palette (matches categories.py group names)
GROUP_COLORS: dict[str, str] = {
    "Sports":        "#4361EE",
    "Politics":      "#EF233C",
    "Finance":       "#06D6A0",
    "Crypto":        "#F9C74F",
    "Entertainment": "#8338EC",
    "Weather":       "#48CAE4",
    "Media":         "#ADB5BD",
    "World Events":  "#FB5607",
    "Science/Tech":  "#3A0CA3",
    "Esports":       "#52B788",
    "Other":         "#6C757D",
}

# ─── rcParams dict ────────────────────────────────────────────────────────────
_RC: dict[str, Any] = {
    "figure.facecolor":   "white",
    "figure.dpi":         150,
    "axes.facecolor":     "white",
    "axes.grid":          True,
    "grid.color":         GRID_C,
    "grid.linewidth":     0.55,
    "grid.alpha":         1.0,
    "axes.axisbelow":     True,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.7,
    "axes.edgecolor":     "#ADB5BD",
    "axes.labelpad":      6,
    "axes.titlepad":      10,
    "font.family":        "sans-serif",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.titleweight":   "semibold",
    "axes.labelsize":     11,
    "xtick.labelsize":    9.5,
    "ytick.labelsize":    9.5,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "xtick.major.width":  0.7,
    "ytick.major.width":  0.7,
    "legend.fontsize":    9.5,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   GRID_C,
    "legend.borderpad":   0.5,
    "lines.linewidth":    1.8,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.1,
}


@contextlib.contextmanager
def paper_style():
    """Context manager: apply paper rcParams for the duration of the block."""
    with mpl.rc_context(_RC):
        yield


def new_fig(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    suptitle: str = "",
    **kwargs: Any,
) -> tuple[plt.Figure, Any]:
    """Create a figure with paper style applied. Returns (fig, axes)."""
    if figsize is None:
        figsize = (7.2 * ncols, 5.0 * nrows)
    with mpl.rc_context(_RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="semibold", y=1.02)
    return fig, axes


# ─── Axis helpers ─────────────────────────────────────────────────────────────

def clean_ax(
    ax: plt.Axes,
    xlabel: str = "",
    ylabel: str = "",
    title:  str = "",
    zero_h: bool = True,
    zero_v: bool = False,
) -> plt.Axes:
    """Apply consistent axis styling: labels, reference lines, ticks."""
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontweight="semibold")
    if zero_h:
        ax.axhline(0, color=GRAY, lw=0.85, linestyle="--", alpha=0.65, zorder=0)
    if zero_v:
        ax.axvline(0, color=GRAY, lw=0.85, linestyle="--", alpha=0.65, zorder=0)
    ax.tick_params(length=3)
    return ax


def stat_box(ax: plt.Axes, text: str, loc: str = "upper right") -> None:
    """Annotate an axis with a stats box (p-value, t-stat, etc.)."""
    corners: dict[str, dict[str, Any]] = {
        "upper right": {"x": 0.97, "y": 0.97, "ha": "right", "va": "top"},
        "upper left":  {"x": 0.03, "y": 0.97, "ha": "left",  "va": "top"},
        "lower right": {"x": 0.97, "y": 0.03, "ha": "right", "va": "bottom"},
        "lower left":  {"x": 0.03, "y": 0.03, "ha": "left",  "va": "bottom"},
    }
    kw = corners.get(loc, corners["upper right"])
    ax.text(
        kw["x"], kw["y"], text,
        transform=ax.transAxes,
        ha=kw["ha"], va=kw["va"],
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor=GRID_C,
            linewidth=0.8,
            alpha=0.95,
        ),
    )


def sig_stars(p: float) -> str:
    """Return significance stars for a p-value."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def shade_h(
    ax: plt.Axes,
    xmin: float,
    xmax: float,
    color: str = AMBER,
    alpha: float = 0.10,
    label: str = "",
) -> None:
    """Shade a vertical span (e.g., longshot tail zone)."""
    ax.axvspan(xmin, xmax, color=color, alpha=alpha, lw=0, zorder=0,
               label=label if label else None)


def shade_v(
    ax: plt.Axes,
    ymin: float,
    ymax: float,
    color: str = AMBER,
    alpha: float = 0.10,
    label: str = "",
) -> None:
    """Shade a horizontal span (e.g., confidence region)."""
    ax.axhspan(ymin, ymax, color=color, alpha=alpha, lw=0, zorder=0,
               label=label if label else None)


def bar_colors(values, pos_color: str = GREEN, neg_color: str = RED) -> list[str]:
    """Return a list of colours: green for positive, red for negative."""
    return [pos_color if v >= 0 else neg_color for v in values]


def group_color(group_name: str) -> str:
    """Return the colour for a category group."""
    return GROUP_COLORS.get(group_name, GRAY)
