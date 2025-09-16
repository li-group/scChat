"""
Seurat-like palettes for Python (matplotlib/seaborn).

Notes:
- Discrete palette approximates ggplot2/Seurat default distinct hues using
  seaborn's "husl"/"hls" space for even hue spacing.
- Continuous palette mimics Seurat's FeaturePlot default of lightgrey → blue.
- Diverging heatmap palette provides a blue–white–red scheme often seen in
  Seurat heatmaps (positive vs negative values). If data are strictly
  positive, consider the continuous palette instead.
"""

from __future__ import annotations

from typing import List

import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def seurat_discrete(n: int) -> List:
    """Return n distinct colors similar to Seurat/ggplot2 default hues.

    Uses seaborn's "husl" palette (uniformly spaced hues in HCL space).
    """
    # husl is the newer name; seaborn also accepts "hls". husl tends to look
    # closer to ggplot2 hue defaults.
    return sns.color_palette("husl", n)


def seurat_continuous_lightgrey_blue() -> LinearSegmentedColormap:
    """Continuous colormap mimicking Seurat FeaturePlot default.

    Low: lightgrey, High: blue.
    """
    return LinearSegmentedColormap.from_list(
        "seurat_featureplot",
        ["#D3D3D3", "#377EB8"],  # lightgrey → blue (ggplot2-ish blue)
        N=256,
    )


def seurat_diverging_bwr() -> LinearSegmentedColormap:
    """Diverging blue–white–red heatmap colormap.

    Useful for centered data (e.g., z-scored expression). Approximates
    typical Seurat heatmap aesthetics when values span negative→positive.
    """
    return LinearSegmentedColormap.from_list(
        "seurat_bwr",
        ["#3B4CC0", "#FFFFFF", "#B40426"],  # blue → white → red
        N=256,
    )


def seurat_continuous_lightgrey_red() -> LinearSegmentedColormap:
    """Continuous colormap: lightgrey → red (ggplot2/Seurat-like red).

    Useful when emphasizing magnitude with a single hue.
    """
    return LinearSegmentedColormap.from_list(
        "seurat_lightgrey_red",
        ["#D3D3D3", "#E41A1C"],  # lightgrey → red (Set1 red)
        N=256,
    )


def seurat_blue_to_red() -> LinearSegmentedColormap:
    """Sequential blue→red colormap (no white midpoint).

    Ideal for mapping increasing significance (e.g., -log10 p) to warmer hues.
    """
    return LinearSegmentedColormap.from_list(
        "seurat_blue_red",
        ["#3B4CC0", "#B40426"],  # blue → red
        N=256,
    )


def seurat_blue_to_lightred() -> LinearSegmentedColormap:
    """Sequential blue→light red colormap.

    Uses ggplot2/Seurat-like light red (≈ #F8766D) for the high end.
    """
    return LinearSegmentedColormap.from_list(
        "seurat_blue_lightred",
        ["#3B4CC0", "#F8766D"],  # blue → light red
        N=256,
    )
