"""Colour scales of the C++ LaNet-vi (``types.cpp``, ``graphics_kcores.cpp``).

The rainbow runs magenta -> blue -> cyan -> green -> yellow -> red, so the maximum index
is red; the black-and-white scale runs white -> grey -> black, so the maximum index is
black. The stop lists are the C++ ones verbatim: some stops lie outside [0, 1]
(``white`` is ``2.0``, the ``blackwhite`` positions are ``-0.75`` and ``1.23``) so that
the extreme indices saturate; the results are clamped as the SVG writer did.
"""

import matplotlib.colors as mcolors

from lanet_vi.models.config import BackgroundColor, ColorScheme

RGB = tuple[float, float, float]

# Rainbow colours (types.cpp)
MAGENTA: RGB = (1.0, 0.2, 1.0)
BLUE: RGB = (0.2, 0.2, 1.0)
CYAN: RGB = (0.2, 1.0, 1.0)
GREEN: RGB = (0.2, 1.0, 0.2)
YELLOW: RGB = (1.0, 1.0, 0.0)
RED: RGB = (1.0, 0.2, 0.2)

# B&W scale (types.cpp); white is deliberately over-bright
WHITE: RGB = (2.0, 2.0, 2.0)
GRAY: RGB = (0.7, 0.7, 0.7)
BLACK: RGB = (0.0, 0.0, 0.0)

#: Colour stops for ``col`` images (``rainbow`` in types.cpp)
RAINBOW: list[tuple[RGB, float]] = [
    (MAGENTA, 0.01),
    (BLUE, 0.20),
    (CYAN, 0.35),
    (GREEN, 0.50),
    (YELLOW, 0.65),
    (RED, 0.97),
]

#: Colour stops for ``bw`` and ``bwi`` images (``blackwhite`` in types.cpp)
BLACKWHITE: list[tuple[RGB, float]] = [
    (WHITE, -0.75),
    (GRAY, 0.41),
    (BLACK, 1.23),
]


def get_color_scale(scheme: ColorScheme) -> list[tuple[RGB, float]]:
    """
    Get the colour stops of a colour scheme.

    Parameters
    ----------
    scheme : ColorScheme
        Colour scheme to use

    Returns
    -------
    list[tuple[RGB, float]]
        ``(colour, position)`` stops; ``bw`` and ``bwi`` share the same list (the
        interlacing happens in the position, not in the stops)
    """
    if scheme == ColorScheme.COLOR:
        return list(RAINBOW)
    return list(BLACKWHITE)


def interpolate_color(color1: RGB, color2: RGB, alpha: float) -> RGB:
    """
    Linearly interpolate between two colours.

    Parameters
    ----------
    color1 : RGB
        First colour (R, G, B)
    color2 : RGB
        Second colour (R, G, B)
    alpha : float
        Interpolation factor (0.0 = color1, 1.0 = color2)

    Returns
    -------
    RGB
        Interpolated colour
    """
    r = color1[0] * (1.0 - alpha) + color2[0] * alpha
    g = color1[1] * (1.0 - alpha) + color2[1] * alpha
    b = color1[2] * (1.0 - alpha) + color2[2] * alpha
    return (r, g, b)


def clamp_color(color: RGB) -> RGB:
    """Clamp every channel to ``[0, 1]`` (the SVG writer saturated at 255)."""
    return (
        min(max(color[0], 0.0), 1.0),
        min(max(color[1], 0.0), 1.0),
        min(max(color[2], 0.0), 1.0),
    )


def scale_color(color: RGB, factor: float) -> RGB:
    """Multiply every channel by ``factor`` and clamp (edge and luminosity shading)."""
    return clamp_color((color[0] * factor, color[1] * factor, color[2] * factor))


def default_node_color(background: BackgroundColor) -> RGB:
    """Colour of a node absent from the colours file: white on black, black on white."""
    return (1.0, 1.0, 1.0) if background == BackgroundColor.BLACK else (0.0, 0.0, 0.0)


def compute_shell_color(
    shell_index: int,
    max_shell_index: int,
    color_scheme: ColorScheme,
    color_scale_max: int | None = None,
    *,
    background: BackgroundColor = BackgroundColor.BLACK,
    dense: bool = False,
) -> RGB:
    """
    Compute the colour of a shell or dense index (``computeHostColorByShellIndex``).

    Parameters
    ----------
    shell_index : int
        Shell or dense index of the node
    max_shell_index : int
        Maximum shell/dense index in the network
    color_scheme : ColorScheme
        Colour scheme to use
    color_scale_max : int | None
        Index shown with the last colour of the scale (``-colorScaleMaxValue``); higher
        indices get the same colour. Defaults to ``max_shell_index``.
    background : BackgroundColor
        Background of the picture; only k-dense colours depend on it
    dense : bool
        Apply the k-dense rules of ``graphics_kdenses.cpp``: on a black background the
        luminosity alternates as for k-cores, on a white background it is a constant 0.9

    Returns
    -------
    RGB
        RGB colour, every channel in ``[0, 1]``

    Notes
    -----
    Position on the scale is ``(min(i, max) - 1) / (max - 1)`` for ``col`` and ``bw``;
    ``bwi`` interlaces even and odd indices over the two halves of the scale. A single
    shell (``max == 1``) is red. With ``col``, consecutive shells alternate a luminosity
    of 0.7 and 1.2 so neighbouring rings stay distinguishable.

    Examples
    --------
    >>> compute_shell_color(5, 5, ColorScheme.COLOR)  # top shell: red, luminosity 1.2
    (1.0, 0.24, 0.24)
    """
    color_list = get_color_scale(color_scheme)
    max_value = color_scale_max if color_scale_max is not None else max_shell_index
    clamped_shell = min(shell_index, max_value)

    # Position on the scale
    if max_value <= 1:
        base_color = RED
    else:
        interlaced = color_scheme == ColorScheme.GRAYSCALE_INTERLACED
        if interlaced and (clamped_shell + max_value) % 2 == 0:
            position = ((max_value - 1) + (clamped_shell - 1)) / (2.0 * (max_value - 1))
        elif interlaced:
            position = (clamped_shell - 1) / (2.0 * (max_value - 1))
        else:
            position = (clamped_shell - 1) / (max_value - 1)

        # Colour calculation: constant outside the stops, linear between them
        first_color, first_pos = color_list[0]
        last_color, last_pos = color_list[-1]
        if position < first_pos:
            base_color = first_color
        elif position > last_pos:
            base_color = last_color
        else:
            base_color = last_color
            for (c1, p1), (c2, p2) in zip(color_list, color_list[1:], strict=False):
                if p1 <= position <= p2:
                    base_color = interpolate_color(c1, c2, (position - p1) / (p2 - p1))
                    break

    # Luminosity alternation between consecutive shells (col only)
    if color_scheme == ColorScheme.COLOR:
        if dense and background == BackgroundColor.WHITE:
            luminosity = 0.9  # graphics_kdenses.cpp: no alternation on white
        else:
            luminosity = 0.7 + 0.5 * (clamped_shell % 2)
        return scale_color(base_color, luminosity)
    return clamp_color(base_color)


def create_matplotlib_colormap(
    color_scheme: ColorScheme,
    n_colors: int = 256,
) -> mcolors.LinearSegmentedColormap:
    """
    Create a matplotlib colormap from a colour scheme.

    Parameters
    ----------
    color_scheme : ColorScheme
        Colour scheme to use
    n_colors : int
        Number of discrete colours in the colormap

    Returns
    -------
    mcolors.LinearSegmentedColormap
        Matplotlib colormap over ``[0, 1]``; stops outside that range are clipped to
        it, so the map starts and ends on the saturated colours
    """
    stops = get_color_scale(color_scheme)
    first, last = stops[0][1], stops[-1][1]
    positions = [min(max((p - first) / (last - first), 0.0), 1.0) for _, p in stops]
    colors = [clamp_color(c) for c, _ in stops]
    return mcolors.LinearSegmentedColormap.from_list(
        f"lanet_{color_scheme.value}", list(zip(positions, colors, strict=True)), N=n_colors
    )
