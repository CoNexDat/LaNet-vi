"""Matplotlib-based renderer for network visualization.

The picture is laid out as the C++ SVG writer did (``svg.cpp``, ``graphics_kcores.cpp``
``generateNetworkFile``): the viewport is the layout's frame, scaled uniformly to the
requested pixel size and centered; edges are drawn under the nodes in increasing index
order; the color and degree legends are drawn in layout units in the margins of the
frame so they scale with the picture.
"""

import math
import random
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import EllipseCollection, LineCollection

from lanet_vi.community.base import CommunityResult
from lanet_vi.decomposition.kdenses import MIN_DENSE_INDEX
from lanet_vi.models.config import (
    BackgroundColor,
    CommunityConfig,
    MeasureType,
    VisualizationConfig,
)
from lanet_vi.models.graph import DecompositionResult, VisualizationLayout
from lanet_vi.visualization.colors import compute_shell_color
from lanet_vi.visualization.community_viz import (
    draw_community_boundaries,
    draw_community_circles,
)
from lanet_vi.visualization.lanet_layout import RadiusLaw, strength_radii

#: Legend title per decomposition type (``m-core`` for k-dense with ``-measure mcore``)
_LEGEND_TITLES = {"kcores": "k-core", "kdenses": "k-dense", "dcores": "d-core"}

#: Figure DPI; the pixel size is ``width x height`` exactly
_DPI = 100


def render_network(
    graph: nx.Graph,
    layout: VisualizationLayout,
    decomposition: DecompositionResult,
    config: VisualizationConfig,
    output_path: Path | str,
    node_names: dict[int, str] | None = None,
    *,
    custom_colors: bool = False,
    measure: MeasureType = MeasureType.MCORE,
    communities: CommunityResult | None = None,
    community_config: CommunityConfig | None = None,
) -> None:
    """
    Render network visualization using matplotlib.

    Parameters
    ----------
    graph : nx.Graph
        Network graph
    layout : VisualizationLayout
        Layout with node positions and visual properties
    decomposition : DecompositionResult
        Decomposition results
    config : VisualizationConfig
        Visualization configuration
    output_path : Union[Path, str]
        Output file path (.png, .pdf, .svg)
    node_names : Optional[Dict[int, str]]
        Optional node names for labels
    custom_colors : bool
        Nodes were colored from a colors file: the color legend is not drawn (the C++
        hid it with ``-colorsFile``)
    measure : MeasureType
        Labels of the k-dense legend: ``mcore`` prints ``k - 2``, ``kdense`` prints ``k``
    communities : Optional[CommunityResult]
        Communities to outline under the network: a translucent convex hull
        (``community_config.draw_boundaries``) and/or circle (``draw_circles``) per
        community, in the community's color
    community_config : Optional[CommunityConfig]
        Which overlays to draw and how (defaults when None)

    Examples
    --------
    >>> render_network(G, layout, decomp, config, "output.png")
    """
    output_path = Path(output_path)
    background = "white" if config.background == BackgroundColor.WHITE else "black"

    # The figure is exactly width x height pixels and the axes fill it
    fig = plt.figure(figsize=(config.width / _DPI, config.height / _DPI), dpi=_DPI)
    fig.patch.set_facecolor(background)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_facecolor(background)
    ax.set_aspect("equal")
    ax.axis("off")

    # Viewport: the frame scaled uniformly to fit the picture and centered (the SVG
    # default preserveAspectRatio "meet"), so the pixel size never distorts the layout
    xmin, xmax, ymin, ymax = layout.bounds
    px_per_unit = min(config.width / (xmax - xmin), config.height / (ymax - ymin))
    half_w = config.width / px_per_unit / 2.0
    half_h = config.height / px_per_unit / 2.0
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    ax.set_xlim(cx - half_w, cx + half_w)
    ax.set_ylim(cy - half_h, cy + half_h)
    pts_per_unit = px_per_unit * 72.0 / _DPI

    if config.draw_circles:
        _draw_component_circles(ax, layout, config)

    if communities is not None:
        _draw_communities(ax, layout, communities, community_config or CommunityConfig())

    _draw_edges(ax, layout, config, decomposition, px_per_unit)
    _draw_nodes(ax, layout, config, px_per_unit)

    if node_names:
        _draw_labels(ax, layout, node_names, config, decomposition)

    if config.show_color_legend and not custom_colors:
        _draw_degree_scale(ax, decomposition, config, layout.frame, pts_per_unit, measure)

    if config.show_degree_scale:
        _draw_size_legend(ax, graph, config, layout, px_per_unit)

    plt.savefig(output_path, dpi=_DPI, facecolor=background)
    plt.close(fig)


def _draw_component_circles(
    ax: plt.Axes,
    layout: VisualizationLayout,
    config: VisualizationConfig,
) -> None:
    """Draw border circles for components."""
    edge_color = "black" if config.background == BackgroundColor.WHITE else "white"

    for comp in layout.components:
        if comp.center and comp.radius:
            circle = mpatches.Circle(
                comp.center,
                radius=comp.radius,
                fill=False,
                edgecolor=edge_color,
                linewidth=0.5,
                alpha=0.3,
            )
            ax.add_patch(circle)


def _draw_communities(
    ax: plt.Axes,
    layout: VisualizationLayout,
    communities: CommunityResult,
    config: CommunityConfig,
) -> None:
    """Outline the communities under the edges: convex hulls and/or bounding circles."""
    if config.draw_boundaries:
        draw_community_boundaries(
            ax,
            communities,
            layout.node_positions,
            alpha=config.boundary_alpha,
            colormap=config.colormap,
            zorder=0,
        )
    if config.draw_circles:
        draw_community_circles(
            ax,
            communities,
            layout.node_positions,
            alpha=config.boundary_alpha,
            colormap=config.colormap,
            zorder=0,
        )


def _draw_edges(
    ax: plt.Axes,
    layout: VisualizationLayout,
    config: VisualizationConfig,
    decomposition: DecompositionResult,
    px_per_unit: float,
) -> None:
    """Draw the visible edges under the nodes, lowest index first.

    Each edge is two half-segments with their own color (the C++ two cylinders meeting
    at the midpoint); widths are in layout units and never thinner than a pixel.
    """
    text_color = "black" if config.background == BackgroundColor.WHITE else "white"
    positions = layout.node_positions
    edges = [(u, v) for u, v in layout.visible_edges if u in positions and v in positions]
    if not edges:
        return

    if not (config.gradient_edges and layout.edge_colors):
        # Plain edges in the text color
        segments = [[positions[u], positions[v]] for u, v in edges]
        ax.add_collection(
            LineCollection(
                segments, colors=text_color, linewidths=0.5, alpha=config.opacity, zorder=1
            )
        )
        return

    # Draw order: increasing edge index, so the core's edges end up on top. The C++
    # painted in call order (edges between clusters, then inside clusters, walking the
    # component tree outside-in), which the index order approximates.
    edge_indices = decomposition.metadata.get("edge_indices")
    node_index = decomposition.node_indices

    def index_of(u: int, v: int) -> int:
        if edge_indices is not None:
            return int(edge_indices.get((u, v) if u < v else (v, u), MIN_DENSE_INDEX))
        return min(node_index.get(u, 0), node_index.get(v, 0))

    edges.sort(key=lambda e: index_of(*e))

    min_width = 1.0 / px_per_unit  # one pixel, in layout units
    pts_per_unit = px_per_unit * 72.0 / _DPI
    gray = (0.5, 0.5, 0.5)
    segments = []
    colors = []
    widths = []
    for u, v in edges:
        (x1, y1), (x2, y2) = positions[u], positions[v]
        mid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        color_u, color_v = layout.edge_colors.get((u, v), (gray, gray))
        width = max(layout.edge_widths.get((u, v), 0.0), min_width) * pts_per_unit
        segments.append([(x1, y1), mid])
        colors.append(color_u)
        widths.append(width)
        segments.append([mid, (x2, y2)])
        colors.append(color_v)
        widths.append(width)

    ax.add_collection(
        LineCollection(
            segments,
            colors=colors,
            linewidths=widths,
            alpha=config.opacity,
            capstyle="round",
            zorder=1,
        )
    )


def _draw_nodes(
    ax: plt.Axes,
    layout: VisualizationLayout,
    config: VisualizationConfig,
    px_per_unit: float,
) -> None:
    """Draw network nodes as opaque circles in layout units, never smaller than a pixel."""
    edge_color = config.node_edge_color if config.node_edge_color else "none"

    # Radii come from the layout (the C++ computeHostRatio: 0.4 units for the largest
    # degree, one unit between shells). With many shells that is a fraction of a pixel,
    # so every node keeps at least a one-pixel radius, as the ray-traced spheres did.
    min_radius = 1.0 / px_per_unit

    def radius_of(node: int) -> float:
        return max(layout.node_sizes.get(node, 0.0), min_radius)

    # For large graphs, one collection instead of individual patches (much faster).
    # EllipseCollection with units="xy" keeps the radii in data units, like the patches.
    if len(layout.node_positions) > 1000:
        offsets = []
        colors = []
        diameters = []
        for node, (x, y) in layout.node_positions.items():
            offsets.append((x, y))
            colors.append(layout.node_colors.get(node, (0.7, 0.7, 0.7)))
            diameters.append(2.0 * radius_of(node))
        collection = EllipseCollection(
            diameters,
            diameters,
            np.zeros(len(diameters)),
            units="xy",
            offsets=offsets,
            offset_transform=ax.transData,
            facecolors=colors,
            edgecolors=edge_color,
            linewidths=0.3,
            zorder=2,
        )
        ax.add_collection(collection)
    else:
        # For small graphs, use individual patches for better quality
        for node, (x, y) in layout.node_positions.items():
            color = layout.node_colors.get(node, (0.7, 0.7, 0.7))

            circle = mpatches.Circle(
                (x, y),
                radius=radius_of(node),
                facecolor=color,
                edgecolor=edge_color,
                linewidth=0.5,
                zorder=2,
            )
            ax.add_patch(circle)


def _draw_labels(
    ax: plt.Axes,
    layout: VisualizationLayout,
    node_names: dict[int, str],
    config: VisualizationConfig,
    decomposition: DecompositionResult,
) -> None:
    """Draw node labels with optional filtering by k-core."""
    if not config.show_node_labels:
        return

    text_color = "black" if config.background == BackgroundColor.WHITE else "white"

    for node, (x, y) in layout.node_positions.items():
        if node not in node_names:
            continue

        # Apply k-core filtering if not labeling all nodes
        if not config.label_all_nodes:
            node_kcore = decomposition.node_indices.get(node, 0)

            # Check k-core range
            if config.label_kcore_min is not None and node_kcore < config.label_kcore_min:
                continue
            if config.label_kcore_max is not None and node_kcore > config.label_kcore_max:
                continue

        ax.text(
            x,
            y,
            node_names[node],
            fontsize=8 * config.font_zoom,
            ha="center",
            va="center",
            color=text_color,
            zorder=3,
        )


def _legend_fontsize(config: VisualizationConfig, size_units: float, pts_per_unit: float) -> float:
    """Legend font size in points: the configured one, else the C++ size in layout units."""
    if config.legend_fontsize:
        return float(config.legend_fontsize)
    return size_units * pts_per_unit


def _draw_degree_scale(
    ax: plt.Axes,
    decomposition: DecompositionResult,
    config: VisualizationConfig,
    frame: float,
    pts_per_unit: float,
    measure: MeasureType,
) -> None:
    """Draw the color legend: one circle per index, in the right margin of the frame.

    Positions follow ``generateNetworkFile`` (``graphics_kcores.cpp``): a column at
    ``1.125`` frames to the right of the center, from ``-0.9`` frames upwards, with at
    most one label every ``max // 15 + 1`` indices counted from the top; the label is
    written in the index color. The k-dense legend starts at 2 and, with ``mcore``,
    labels each index ``k - 2``; its positions use the k-core formulas too (the
    ``graphics_kdenses.cpp`` variant drops the network radius ``R`` and one ``u`` from
    them, which leaves the legend inside the network on large pictures).
    """
    max_idx = decomposition.max_index
    is_dense = decomposition.decomp_type == "kdenses"
    first = MIN_DENSE_INDEX if is_dense else 1
    if max_idx < first or frame <= 0.0:
        return  # nothing on the scale (edgeless graph)

    # The color scale maximum is given in m-core units for k-dense with -measure mcore
    color_scale_max = config.color_scale_max_value
    if is_dense and color_scale_max is not None and measure == MeasureType.MCORE:
        color_scale_max += 2
    label_offset = 2 if is_dense and measure == MeasureType.MCORE else 0

    # Layout units: R * u is the frame without gamma
    ru = frame / config.gamma
    max_si = max(max_idx, 15)
    x = frame * 27.0 / 24.0
    separation = 0.8 * ru * 2.0 / (max_si * 5.0)
    radius = 1.5 * separation
    step = 5.0 * config.gamma * config.unit_length * separation
    text_size = 4.0 * 0.8 * ru * 2.0 / (15.0 * 5.0)
    fontsize = _legend_fontsize(config, text_size, pts_per_unit)
    label_every = max_idx // 15 + 1

    for i in range(first, max_idx + 1):
        color = compute_shell_color(
            i,
            max_idx,
            config.color_scheme,
            color_scale_max,
            background=config.background,
            dense=is_dense,
        )
        y = -0.9 * frame + step * i
        ax.add_patch(
            mpatches.Circle((x, y), radius=radius, facecolor=color, edgecolor="none", zorder=4)
        )
        if max_idx <= 15 or (max_idx - i) % label_every == 0:
            ax.text(
                x + 3.0 * separation,
                y,
                str(i - label_offset),
                fontsize=fontsize,
                ha="left",
                va="center",
                color=color,
                zorder=4,
            )

    title_color = "black" if config.background == BackgroundColor.WHITE else "white"
    title = _LEGEND_TITLES.get(decomposition.decomp_type, decomposition.decomp_type)
    if label_offset:
        title = "m-core"
    ax.text(
        x,
        -0.9 * frame + step * (max_idx + 1),
        title,
        fontsize=fontsize,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=title_color,
        zorder=4,
    )


def _draw_size_legend(
    ax: plt.Axes,
    graph: nx.Graph,
    config: VisualizationConfig,
    layout: VisualizationLayout,
    px_per_unit: float,
) -> None:
    """Draw the degree legend: up to five sample nodes in the left margin of the frame.

    As ``generateNetworkFile``: degrees ``ceil(dmax / 4**i)`` while above 1, drawn with
    the radius the nodes of that degree have in the picture (``node_radius`` times
    ``node_size_scale``), white on black / gray on white, at ``-1.25`` frames from the
    center. Layouts with strength-based radii show strengths ``smax / 4**i`` instead
    (the C++ placed those without the ``u * R`` factor, which collapses the legend on
    large pictures; the degree spacing is used for both).
    """
    frame = layout.frame
    degrees = dict(graph.degree())
    if not degrees or frame <= 0.0:
        return
    max_degree = max(degrees.values())
    if max_degree < 1:
        return  # nothing to scale on an edgeless graph

    ru = frame / config.gamma
    scale = config.node_size_scale
    min_radius = 1.0 / px_per_unit
    pts_per_unit = px_per_unit * 72.0 / _DPI
    sphere_color = (1.0, 1.0, 1.0) if config.background == BackgroundColor.BLACK else (0.7,) * 3
    text_color = "black" if config.background == BackgroundColor.WHITE else "white"
    x = -frame * 15.0 / 12.0

    max_strength = 0.0
    if layout.weighted:
        max_strength = max(
            (sum(float(d.get("weight", 1.0)) for d in graph[v].values()) for v in graph),
            default=0.0,
        )
    # Same rule as the node radii: strengths only when their law applies
    weighted = strength_radii(layout.weighted, max_strength)
    law = layout.radius_law or RadiusLaw(max_degree, max_strength, weighted)

    samples: list[tuple[str, float]] = []
    if weighted:
        separation = (
            1.5 * ru * (0.0007 + 0.029 * math.log(1 + max_strength) / math.log(max_strength))
        )
        for i in range(5):
            strength = max_strength / 4.0**i
            radius = scale * law(0, strength, weighted=True)
            samples.append((f"{strength:g}", radius))
    else:
        log_max = math.log(max_degree) if max_degree > 1 else 1.0
        separation = 1.5 * ru * (0.0007 + 0.029 * math.log(1 + max_degree) / log_max)
        for i in range(5):
            degree = math.ceil(max_degree / 4.0**i)
            if degree <= 1:
                break
            samples.append((str(degree), scale * law(degree)))

    # The C++ rows are 3 separations apart and the text 2 separations tall, tuned for
    # large classic pictures; on tiny ones, and in the pow / log modes (radii of up to a
    # fifth of the unit disc), the biggest samples would overlap, so a row is pushed up
    # until it clears the previous circle. The font keeps the C++ size.
    fontsize = _legend_fontsize(config, 2.0 * separation, pts_per_unit)
    y = -0.5 * ru
    previous_radius = 0.0
    for label, radius in samples:
        y = max(y + 3.0 * separation, y + previous_radius + radius + separation)
        previous_radius = radius
        ax.add_patch(
            mpatches.Circle(
                (x, y),
                radius=max(radius, min_radius),
                facecolor=sphere_color,
                edgecolor="none",
                zorder=4,
            )
        )
        ax.text(
            x + max(separation, radius + 0.5 * separation),
            y,
            label,
            fontsize=fontsize,
            ha="left",
            va="center",
            color=text_color,
            zorder=4,
        )
    if samples:
        ax.text(
            x,
            max(y + 3.0 * separation, y + previous_radius + separation),
            "degree" if not weighted else "strength",
            fontsize=fontsize,
            fontweight="bold",
            ha="left",
            va="bottom",
            color=text_color,
            zorder=4,
        )


def select_visible_edges(
    graph: nx.Graph,
    config: VisualizationConfig,
    seed: int | None = None,
) -> list[tuple[int, int]]:
    """
    Select the edges to draw: an independent Bernoulli draw per edge, as the C++ did.

    Parameters
    ----------
    graph : nx.Graph
        Network graph
    config : VisualizationConfig
        ``edges_percent`` and ``min_edges``
    seed : int | None
        Seed of the draw, so the same seed gives the same picture

    Returns
    -------
    List[Tuple[int, int]]
        Edges to render, in the graph's order

    Notes
    -----
    Every edge is kept with probability ``max(edges_percent, min_edges / E)``
    (``graphics_kcores.cpp`` ``addCluster``), so about that fraction of the edges is
    drawn, spread over all shells in proportion to their edge counts.
    """
    edges = list(graph.edges())
    num_edges = len(edges)
    if num_edges == 0:
        return []

    probability = max(config.edges_percent, config.min_edges / num_edges)
    if probability >= 1.0:
        return edges
    if probability <= 0.0:
        return []

    rng = random.Random(seed)
    return [edge for edge in edges if rng.random() < probability]
