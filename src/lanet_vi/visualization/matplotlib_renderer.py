"""Matplotlib-based renderer for network visualization."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np

from lanet_vi.models.config import BackgroundColor, VisualizationConfig
from lanet_vi.models.graph import DecompositionResult, VisualizationLayout
from lanet_vi.visualization.colors import compute_shell_color


def render_network(
    graph: nx.Graph,
    layout: VisualizationLayout,
    decomposition: DecompositionResult,
    config: VisualizationConfig,
    output_path: Union[Path, str],
    node_names: Optional[Dict[int, str]] = None,
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

    Examples
    --------
    >>> render_network(G, layout, decomp, config, "output.png")
    """
    output_path = Path(output_path)

    # Create figure
    fig, ax = plt.subplots(
        figsize=(config.width / 100, config.height / 100),
        dpi=100,
        facecolor="white" if config.background == BackgroundColor.WHITE else "black",
    )

    # Set background color
    ax.set_facecolor("white" if config.background == BackgroundColor.WHITE else "black")

    # Get bounds
    xmin, xmax, ymin, ymax = layout.bounds

    # Add some padding
    padding = max(xmax - xmin, ymax - ymin) * 0.1
    ax.set_xlim(xmin - padding, xmax + padding)
    ax.set_ylim(ymin - padding, ymax + padding)
    ax.set_aspect("equal")

    # Remove axes
    ax.axis("off")

    # Draw component circles if requested
    if config.draw_circles:
        _draw_component_circles(ax, layout, config)

    # Draw edges with k-core layering
    _draw_edges(ax, graph, layout, config, decomposition)

    # Draw nodes
    _draw_nodes(ax, layout, config)

    # Draw node labels if names provided
    if node_names:
        _draw_labels(ax, layout, node_names, config, decomposition)

    # Draw degree scale legend if requested
    if config.show_degree_scale:
        _draw_degree_scale(ax, decomposition, config)

    # Draw size legend if requested
    if config.show_size_legend:
        _draw_size_legend(ax, graph, config)

    # Save figure
    plt.savefig(
        output_path,
        dpi=100,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
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


def _draw_edges(
    ax: plt.Axes,
    graph: nx.Graph,
    layout: VisualizationLayout,
    config: VisualizationConfig,
    decomposition: "DecompositionResult",
) -> None:
    """Draw network edges with optional gradient coloring, layered by k-core."""
    edge_color = "black" if config.background == BackgroundColor.WHITE else "white"

    if config.gradient_edges and layout.edge_colors:
        # Use gradient edge rendering with k-core layering
        # Group edge segments by k-core level for proper layering
        from collections import defaultdict
        segments_by_kcore = defaultdict(lambda: {'segments': [], 'colors': [], 'widths': []})

        for source, target in layout.visible_edges:
            if source not in layout.node_positions or target not in layout.node_positions:
                continue

            x1, y1 = layout.node_positions[source]
            x2, y2 = layout.node_positions[target]

            # Midpoint
            mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            # Get edge colors (darkened endpoint colors)
            edge_colors_pair = layout.edge_colors.get((source, target), ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
            color1, color2 = edge_colors_pair

            # Get edge width
            width = layout.edge_widths.get((source, target), 0.5)

            # Determine k-core level for this edge (minimum of endpoints)
            u_kcore = decomposition.node_indices.get(source, 1)
            v_kcore = decomposition.node_indices.get(target, 1)
            min_kcore = min(u_kcore, v_kcore)

            # Add both edge segments to the appropriate k-core group
            # First half: source -> midpoint (color1)
            segments_by_kcore[min_kcore]['segments'].append([(x1, y1), (mid_x, mid_y)])
            segments_by_kcore[min_kcore]['colors'].append(color1)
            segments_by_kcore[min_kcore]['widths'].append(width)

            # Second half: midpoint -> target (color2)
            segments_by_kcore[min_kcore]['segments'].append([(mid_x, mid_y), (x2, y2)])
            segments_by_kcore[min_kcore]['colors'].append(color2)
            segments_by_kcore[min_kcore]['widths'].append(width)

        # Draw edges in k-core order: lower k-core first (background), higher k-core last (foreground)
        # Use negative zorder to ensure all edges are behind nodes (which have zorder=2)
        max_kcore = max(segments_by_kcore.keys()) if segments_by_kcore else 1
        for kcore in sorted(segments_by_kcore.keys()):
            data = segments_by_kcore[kcore]
            if data['segments']:
                # Map k-core to zorder range: [-max_kcore, -1]
                # Lower k-core gets more negative zorder (further back)
                # Higher k-core gets less negative zorder (closer to front, but still behind nodes)
                edge_zorder = kcore - max_kcore - 1
                lc = LineCollection(
                    data['segments'],
                    colors=data['colors'],
                    linewidths=data['widths'],
                    alpha=config.edge_alpha,
                    zorder=edge_zorder,
                )
                ax.add_collection(lc)
    else:
        # Fallback to simple edge rendering (old behavior)
        for source, target in layout.visible_edges:
            if source in layout.node_positions and target in layout.node_positions:
                x1, y1 = layout.node_positions[source]
                x2, y2 = layout.node_positions[target]

                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=edge_color,
                    linewidth=0.5,
                    alpha=config.opacity,
                    zorder=1,
                )


def _draw_nodes(
    ax: plt.Axes,
    layout: VisualizationLayout,
    config: VisualizationConfig,
) -> None:
    """Draw network nodes using scatter plot for performance."""
    edge_color = config.node_edge_color if config.node_edge_color else "none"

    # For large graphs, use scatter plot instead of individual patches (much faster)
    if len(layout.node_positions) > 1000:
        # Prepare arrays for scatter plot
        x_coords = []
        y_coords = []
        colors = []
        sizes = []

        for node, (x, y) in layout.node_positions.items():
            x_coords.append(x)
            y_coords.append(y)
            colors.append(layout.node_colors.get(node, (0.7, 0.7, 0.7)))
            # scatter uses area (s = πr²), so multiply by π and square
            size = layout.node_sizes.get(node, 5.0)
            sizes.append(np.pi * size ** 2)

        # Draw all nodes at once with scatter
        ax.scatter(
            x_coords,
            y_coords,
            s=sizes,
            c=colors,
            edgecolors=edge_color,
            linewidths=0.3,
            zorder=2,
            alpha=0.9,
        )
    else:
        # For small graphs, use individual patches for better quality
        for node, (x, y) in layout.node_positions.items():
            color = layout.node_colors.get(node, (0.7, 0.7, 0.7))
            size = layout.node_sizes.get(node, 5.0)

            circle = mpatches.Circle(
                (x, y),
                radius=size,
                facecolor=color,
                edgecolor=edge_color,
                linewidth=0.5,
                zorder=2,
            )
            ax.add_patch(circle)


def _draw_labels(
    ax: plt.Axes,
    layout: VisualizationLayout,
    node_names: Dict[int, str],
    config: VisualizationConfig,
    decomposition: DecompositionResult,
) -> None:
    """Draw node labels with optional filtering by k-core."""
    if not config.show_node_labels:
        return

    text_color = "black" if config.background == BackgroundColor.WHITE else "white"

    for node, (x, y) in layout.node_positions.items():
        if node not in node_names or node_names[node] == "0":
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


def _draw_degree_scale(
    ax: plt.Axes,
    decomposition: DecompositionResult,
    config: VisualizationConfig,
) -> None:
    """Draw color scale legend for shell/dense indices using circle markers."""
    # Create custom legend showing the shell/dense index color mapping
    legend_elements = []

    max_idx = (
        config.color_scale_max_value
        if config.color_scale_max_value
        else decomposition.max_index
    )

    # Sample more indices for comprehensive legend (matching reference image)
    # Reference shows ~13 values from 1 to max
    indices = np.linspace(1, max_idx, min(13, max_idx), dtype=int)

    # Only label some entries to avoid clutter (every other entry for readability)
    label_interval = max(1, len(indices) // 6)  # Show ~6-7 labels

    for i, idx in enumerate(indices):
        color = compute_shell_color(idx, max_idx, config.color_scheme)

        # Smaller circles to match reference image
        marker_size = 5  # Reduced from 8 for more compact legend

        # Create a circle marker for the legend
        circle = mpatches.Circle(
            (0, 0),  # Position doesn't matter for legend
            radius=marker_size,
            facecolor=color,
            edgecolor="black",
            linewidth=0.5,
        )

        # Selective labeling: only show labels at intervals
        if i % label_interval == 0 or i == len(indices) - 1:
            label = f"{idx}"
        else:
            label = ""  # Empty label for unlabeled entries

        legend_elements.append((circle, label))

    # Create custom legend handler for circles
    from matplotlib.legend_handler import HandlerPatch

    class HandlerCircle(HandlerPatch):
        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
            p = mpatches.Circle(xy=center, radius=orig_handle.radius)
            self.update_prop(p, orig_handle, legend)
            p.set_transform(trans)
            return [p]

    # Calculate font size: auto-scale with diagram size or use manual setting
    if config.legend_fontsize:
        fontsize = config.legend_fontsize
    else:
        # Auto-scale: base size 8 for 800px width, scale linearly
        fontsize = 8 * (config.width / 800)

    # Set text color based on background (white for dark backgrounds)
    text_color = "black" if config.background == BackgroundColor.WHITE else "white"

    legend = ax.legend(
        handles=[h[0] for h in legend_elements],
        labels=[h[1] for h in legend_elements],
        handler_map={mpatches.Circle: HandlerCircle()},
        loc="center right",  # Moved from "upper right" to match reference image
        frameon=False,  # Remove frame completely
        fontsize=fontsize,
        title="k-core",
        labelcolor=text_color,  # Set label text color
        title_fontproperties={'size': fontsize, 'weight': 'bold'},
    )
    # Set title color manually (labelcolor doesn't affect title)
    legend.get_title().set_color(text_color)
    # Keep reference to prevent it being replaced by subsequent legend calls
    ax.add_artist(legend)


def _draw_size_legend(
    ax: plt.Axes,
    graph: nx.Graph,
    config: VisualizationConfig,
) -> None:
    """Draw size legend showing node degree scale."""
    # Get degree statistics
    degrees = dict(graph.degree())
    if not degrees:
        return

    max_degree = max(degrees.values())
    n_nodes = graph.number_of_nodes()

    # Compute sample sizes to show in legend
    # Use same scaling as in network.py compute_layout()
    import math

    if n_nodes > 1000:
        # Log scaling for large graphs
        sample_degrees = [1, max_degree // 4, max_degree // 2, max_degree]
        sample_sizes = [
            0.3 + 2.0 * math.log(1 + deg) / math.log(1 + max_degree)
            for deg in sample_degrees
        ]
    else:
        # Linear scaling for small graphs
        sample_degrees = [1, max_degree // 2, max_degree]
        sample_sizes = [
            0.5 + 3.0 * (deg / max_degree)
            for deg in sample_degrees
        ]

    # Create legend elements
    legend_elements = []

    from matplotlib.legend_handler import HandlerPatch

    class HandlerCircle(HandlerPatch):
        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            # Ensure proper vertical spacing between legend items
            center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
            p = mpatches.Circle(xy=center, radius=orig_handle.radius)
            self.update_prop(p, orig_handle, legend)
            p.set_transform(trans)
            return [p]

    for deg, size in zip(sample_degrees, sample_sizes):
        circle = mpatches.Circle(
            (0, 0),
            radius=size,
            facecolor="gray",
            edgecolor="black",
            linewidth=0.5,
        )
        legend_elements.append((circle, f"{deg}"))

    # Calculate font size: auto-scale with diagram size or use manual setting
    if config.legend_fontsize:
        fontsize = config.legend_fontsize
    else:
        # Auto-scale: base size 8 for 800px width, scale linearly
        fontsize = 8 * (config.width / 800)

    # Set text color based on background (white for dark backgrounds)
    text_color = "black" if config.background == BackgroundColor.WHITE else "white"

    legend = ax.legend(
        handles=[h[0] for h in legend_elements],
        labels=[h[1] for h in legend_elements],
        handler_map={mpatches.Circle: HandlerCircle()},
        loc="upper left",  # Moved from "lower right" to match reference image
        frameon=False,  # Remove frame completely
        fontsize=fontsize,
        title="degree",
        labelspacing=1.5,  # Increase spacing to prevent overlap
        labelcolor=text_color,  # Set label text color
        title_fontproperties={'size': fontsize, 'weight': 'bold'},
    )
    # Set title color manually (labelcolor doesn't affect title)
    legend.get_title().set_color(text_color)


def select_visible_edges(
    graph: nx.Graph,
    config: VisualizationConfig,
    decomposition: DecompositionResult,
) -> List[Tuple[int, int]]:
    """
    Select subset of edges to display based on configuration.

    Parameters
    ----------
    graph : nx.Graph
        Network graph
    config : VisualizationConfig
        Visualization configuration with edge visibility settings
    decomposition : DecompositionResult
        Decomposition results

    Returns
    -------
    List[Tuple[int, int]]
        List of edges to render
    """
    edges = list(graph.edges())

    if config.edges_percent == 0.0 and config.min_edges == 0:
        return []

    # Calculate number of edges to show
    num_edges = len(edges)
    target_edges = max(
        int(num_edges * config.edges_percent),
        min(config.min_edges, num_edges),
    )

    if target_edges >= num_edges:
        return edges

    # Use stratified sampling: select edges from all k-core levels
    # This ensures we show connectivity across the entire network hierarchy
    import random

    # Group edges by their shell connectivity
    from collections import defaultdict
    edges_by_shell = defaultdict(list)

    for edge in edges:
        u, v = edge
        u_idx = decomposition.node_indices.get(u, 0)
        v_idx = decomposition.node_indices.get(v, 0)
        # Use minimum shell index to categorize edge
        min_shell = min(u_idx, v_idx)
        edges_by_shell[min_shell].append(edge)

    # Sample proportionally from each shell level
    selected_edges = []
    total_edges_available = len(edges)

    for shell_idx in sorted(edges_by_shell.keys(), reverse=True):
        shell_edges = edges_by_shell[shell_idx]
        # Sample proportion of edges from this shell
        n_to_sample = min(
            len(shell_edges),
            max(1, int(len(shell_edges) * target_edges / total_edges_available))
        )
        selected_edges.extend(random.sample(shell_edges, n_to_sample))

        if len(selected_edges) >= target_edges:
            break

    # If we still need more edges, add remaining high-priority edges
    if len(selected_edges) < target_edges:
        remaining = set(edges) - set(selected_edges)
        remaining_sorted = sorted(
            remaining,
            key=lambda e: -(decomposition.node_indices.get(e[0], 0) + decomposition.node_indices.get(e[1], 0))
        )
        selected_edges.extend(remaining_sorted[:target_edges - len(selected_edges)])

    return selected_edges[:target_edges]
