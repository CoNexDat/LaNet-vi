"""Color schemes and utilities for network visualization."""

from typing import List, Optional, Tuple

import matplotlib.colors as mcolors
import numpy as np

from lanet_vi.models.config import ColorScheme


# Color definitions (RGB in 0-1 range)
RED = (1.0, 0.0, 0.0)
BLUE = (0.0, 0.0, 1.0)
GREEN = (0.0, 1.0, 0.0)
YELLOW = (1.0, 1.0, 0.0)
CYAN = (0.0, 1.0, 1.0)
MAGENTA = (1.0, 0.0, 1.0)
ORANGE = (1.0, 0.5, 0.0)
PURPLE = (0.5, 0.0, 0.5)
WHITE = (1.0, 1.0, 1.0)
BLACK = (0.0, 0.0, 0.0)


def get_color_scale(scheme: ColorScheme) -> List[Tuple[Tuple[float, float, float], float]]:
    """
    Get color scale for the given color scheme.

    Parameters
    ----------
    scheme : ColorScheme
        Color scheme to use

    Returns
    -------
    List[Tuple[Tuple[float, float, float], float]]
        List of (color, position) tuples where position is in [0, 1]
    """
    if scheme == ColorScheme.COLOR:
        # Rainbow color scale
        return [
            (BLUE, 0.0),
            (CYAN, 0.16),
            (GREEN, 0.33),
            (YELLOW, 0.50),
            (ORANGE, 0.66),
            (RED, 0.83),
            (MAGENTA, 1.0),
        ]
    elif scheme == ColorScheme.GRAYSCALE:
        # Grayscale
        return [
            (BLACK, 0.0),
            (WHITE, 1.0),
        ]
    else:  # GRAYSCALE_INTERLACED
        # Grayscale with interlacing
        return [
            ((0.3, 0.3, 0.3), 0.0),
            ((0.7, 0.7, 0.7), 0.5),
            (WHITE, 1.0),
        ]


def interpolate_color(
    color1: Tuple[float, float, float],
    color2: Tuple[float, float, float],
    alpha: float,
) -> Tuple[float, float, float]:
    """
    Linearly interpolate between two colors.

    Parameters
    ----------
    color1 : Tuple[float, float, float]
        First color (R, G, B)
    color2 : Tuple[float, float, float]
        Second color (R, G, B)
    alpha : float
        Interpolation factor (0.0 = color1, 1.0 = color2)

    Returns
    -------
    Tuple[float, float, float]
        Interpolated color
    """
    r = color1[0] * (1.0 - alpha) + color2[0] * alpha
    g = color1[1] * (1.0 - alpha) + color2[1] * alpha
    b = color1[2] * (1.0 - alpha) + color2[2] * alpha
    return (r, g, b)


def compute_shell_color(
    shell_index: int,
    max_shell_index: int,
    color_scheme: ColorScheme,
    color_scale_max: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Compute color for a node based on its shell/dense index.

    Parameters
    ----------
    shell_index : int
        Shell or dense index of the node
    max_shell_index : int
        Maximum shell/dense index in the network
    color_scheme : ColorScheme
        Color scheme to use
    color_scale_max : Optional[int]
        Maximum value for color scale (for normalization)

    Returns
    -------
    Tuple[float, float, float]
        RGB color tuple

    Examples
    --------
    >>> color = compute_shell_color(3, 5, ColorScheme.COLOR)
    >>> # Returns color for shell 3 out of max 5
    """
    # Get color scale
    color_list = get_color_scale(color_scheme)

    # Determine maximum value for normalization
    if color_scale_max is not None:
        max_value = color_scale_max
    else:
        max_value = max_shell_index

    # Handle single shell case
    if max_value == 1:
        return RED

    # Compute position in [0, 1] range
    clamped_shell = min(shell_index, max_value)

    if color_scheme == ColorScheme.COLOR or color_scheme == ColorScheme.GRAYSCALE:
        position = (clamped_shell - 1) / (max_value - 1)
    else:  # GRAYSCALE_INTERLACED
        # Interlaced pattern based on parity
        if (clamped_shell + max_value) % 2 == 0:
            position = ((max_value - 1) + (clamped_shell - 1)) / (2.0 * (max_value - 1))
        else:
            position = (clamped_shell - 1) / (2.0 * (max_value - 1))

    # Find colors to interpolate between
    color1, pos1 = color_list[0]
    color2, pos2 = color_list[-1]

    if position <= pos1:
        base_color = color1
    elif position >= pos2:
        base_color = color2
    else:
        # Find bracketing colors
        for i in range(len(color_list) - 1):
            c1, p1 = color_list[i]
            c2, p2 = color_list[i + 1]
            if p1 <= position <= p2:
                alpha = (position - p1) / (p2 - p1)
                base_color = interpolate_color(c1, c2, alpha)
                break

    # Apply luminosity variation for color scheme
    if color_scheme == ColorScheme.COLOR:
        luminosity = 0.7 + 0.3 * (clamped_shell % 2)  # Max 1.0 instead of 1.2
        r = min(base_color[0] * luminosity, 1.0)
        g = min(base_color[1] * luminosity, 1.0)
        b = min(base_color[2] * luminosity, 1.0)
        return (r, g, b)

    return base_color


def create_matplotlib_colormap(
    color_scheme: ColorScheme,
    n_colors: int = 256,
) -> mcolors.LinearSegmentedColormap:
    """
    Create a matplotlib colormap from a color scheme.

    Parameters
    ----------
    color_scheme : ColorScheme
        Color scheme to use
    n_colors : int
        Number of discrete colors in the colormap

    Returns
    -------
    mcolors.LinearSegmentedColormap
        Matplotlib colormap
    """
    color_list = get_color_scale(color_scheme)
    colors = [c for c, _ in color_list]
    positions = [p for _, p in color_list]

    # Create colormap
    cmap = mcolors.LinearSegmentedColormap.from_list(
        f"lanet_{color_scheme.value}", list(zip(positions, colors)), N=n_colors
    )

    return cmap
