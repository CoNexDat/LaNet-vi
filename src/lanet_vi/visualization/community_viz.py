"""Community visualization utilities.

This module provides functions for visualizing network communities, including:
- Coloring nodes by community membership
- Drawing community boundaries
- Creating community-based color palettes
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull, QhullError

from lanet_vi.community.base import CommunityResult
from lanet_vi.logging_config import get_logger

logger = get_logger(__name__)


#: Colormaps with at most this many entries are qualitative: their entries are used one
#: by one instead of sampling the [0, 1] range
_QUALITATIVE_MAX_ENTRIES = 40


def get_community_colors(
    num_communities: int,
    colormap: str = "tab20",
) -> list[tuple[float, float, float]]:
    """Generate distinct colors for communities.

    Parameters
    ----------
    num_communities : int
        Number of communities to generate colors for
    colormap : str
        Matplotlib colormap name (default: "tab20", 20 distinct colors)

    Returns
    -------
    List[Tuple[float, float, float]]
        List of RGB color tuples, one per community id (0-based)

    Notes
    -----
    A qualitative colormap (``tab10``, ``tab20``, ``Set3``, ...) is used entry by entry,
    so up to its size no two communities share a color; with more communities than
    entries the colors are spread evenly over ``hsv`` instead. A continuous colormap
    (``viridis``, ``hsv``, ...) is sampled evenly over its range. The default ``tab20``
    alternates a dark and a light shade of each hue, so up to 10 communities take the
    dark shades (``tab10``) and stay telling apart.
    """
    if num_communities <= 0:
        return []

    if colormap == "tab20" and num_communities <= 10:
        colormap = "tab10"
    cmap = plt.get_cmap(colormap)
    if isinstance(cmap, ListedColormap) and cmap.N <= _QUALITATIVE_MAX_ENTRIES:
        if num_communities <= cmap.N:
            samples = [cmap(i) for i in range(num_communities)]
        else:
            logger.debug(
                f"{num_communities} communities exceed the {cmap.N} colors of {colormap!r}; "
                "spreading them over 'hsv'"
            )
            hsv = plt.get_cmap("hsv")
            samples = [hsv(i / num_communities) for i in range(num_communities)]
    else:
        samples = [cmap(i / num_communities) for i in range(num_communities)]

    return [(float(r), float(g), float(b)) for r, g, b, _alpha in samples]


def _colors_by_community_id(
    community_result: CommunityResult, colormap: str
) -> dict[int, tuple[float, float, float]]:
    """One color per community, keyed by community id (ids need not be 0..n-1)."""
    palette = get_community_colors(len(community_result.communities), colormap=colormap)
    return {
        community.id: color
        for community, color in zip(community_result.communities, palette, strict=True)
    }


def assign_node_colors_by_community(
    community_result: CommunityResult,
    node_positions: dict[int, tuple[float, float]],
    colormap: str = "tab20",
) -> dict[int, tuple[float, float, float]]:
    """Assign colors to nodes based on their community membership.

    Parameters
    ----------
    community_result : CommunityResult
        Community detection result
    node_positions : Dict[int, Tuple[float, float]]
        Node positions (used to determine which nodes to color)
    colormap : str
        Matplotlib colormap name

    Returns
    -------
    Dict[int, Tuple[float, float, float]]
        Mapping from node ID to RGB color tuple

    Examples
    --------
    >>> from lanet_vi.community import detect_communities_louvain
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> communities = detect_communities_louvain(G)
    >>> positions = nx.spring_layout(G)
    >>> colors = assign_node_colors_by_community(communities, positions)
    """
    logger.debug(
        f"Assigning colors to {len(node_positions)} nodes across "
        f"{community_result.num_communities} communities"
    )

    community_colors = _colors_by_community_id(community_result, colormap)

    # Assign colors to nodes
    node_colors = {}
    for node in node_positions:
        comm_id = community_result.get_node_community(node)
        if comm_id is not None and comm_id in community_colors:
            node_colors[node] = community_colors[comm_id]
        else:
            # Default gray for nodes not in any community
            node_colors[node] = (0.7, 0.7, 0.7)

    return node_colors


def _convex_hull(points: np.ndarray) -> np.ndarray | None:
    """Vertices of the convex hull of ``points`` (n x 2), or None if Qhull cannot build one.

    Collinear points have no hull in the strict sense; Qhull's joggle option perturbs
    them into a sliver polygon so the community still shows.
    """
    for options in ("", "QJ"):
        try:
            hull = ConvexHull(points, qhull_options=options)
        except QhullError:
            continue
        vertices: np.ndarray = points[hull.vertices]
        return vertices
    return None


def draw_community_boundaries(
    ax: plt.Axes,
    community_result: CommunityResult,
    node_positions: dict[int, tuple[float, float]],
    alpha: float = 0.2,
    linewidth: float = 2.0,
    colormap: str = "tab20",
    zorder: float = 0.0,
) -> None:
    """Draw convex hull boundaries around communities.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to draw on
    community_result : CommunityResult
        Community detection result
    node_positions : Dict[int, Tuple[float, float]]
        Node positions in 2D space
    alpha : float
        Transparency of the boundary fill (default: 0.2)
    linewidth : float
        Width of the boundary line (default: 2.0)
    colormap : str
        Matplotlib colormap name
    zorder : float
        Drawing order of the hulls (default 0: behind the edges and nodes)

    Notes
    -----
    Only draws boundaries for communities with 3 or more nodes (required for convex hull).
    """
    logger.debug(f"Drawing boundaries for {community_result.num_communities} communities")

    community_colors = _colors_by_community_id(community_result, colormap)

    patches = []
    colors = []

    for community in community_result.communities:
        # Distinct positions of the community's nodes (nodes of one cluster can share a
        # position; a hull needs three distinct points)
        points = sorted(
            {node_positions[node] for node in community.nodes if node in node_positions}
        )
        if len(points) < 3:
            continue

        hull_points = _convex_hull(np.array(points))
        if hull_points is None:
            logger.warning(f"Could not compute the convex hull of community {community.id}")
            continue
        patches.append(Polygon(hull_points, closed=True))
        colors.append(community_colors[community.id])

    # Draw all patches
    if patches:
        collection = PatchCollection(
            patches,
            facecolors=colors,
            alpha=alpha,
            edgecolors=colors,
            linewidths=linewidth,
            zorder=zorder,
        )
        ax.add_collection(collection)
        logger.debug(f"Drew {len(patches)} community boundaries")


def draw_community_circles(
    ax: plt.Axes,
    community_result: CommunityResult,
    node_positions: dict[int, tuple[float, float]],
    padding: float = 0.1,
    alpha: float = 0.15,
    linewidth: float = 2.0,
    colormap: str = "tab20",
) -> None:
    """Draw circles around communities based on their bounding box.

    This is an alternative to convex hulls that works better for small communities.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to draw on
    community_result : CommunityResult
        Community detection result
    node_positions : Dict[int, Tuple[float, float]]
        Node positions in 2D space
    padding : float
        Extra padding around community as fraction of radius (default: 0.1)
    alpha : float
        Transparency of the circle fill (default: 0.15)
    linewidth : float
        Width of the circle line (default: 2.0)
    colormap : str
        Matplotlib colormap name
    """
    logger.debug(f"Drawing circles for {community_result.num_communities} communities")

    community_colors = _colors_by_community_id(community_result, colormap)

    for community in community_result.communities:
        # Get positions of nodes in this community
        points = []
        for node in community.nodes:
            if node in node_positions:
                points.append(node_positions[node])

        if len(points) < 1:
            continue

        points_array = np.array(points)

        # Compute center and radius
        center = points_array.mean(axis=0)
        max_dist = float(np.max(np.linalg.norm(points_array - center, axis=1)))
        radius = max_dist * (1.0 + padding)

        # Draw circle
        color = community_colors[community.id]
        circle = plt.Circle(
            center,
            radius,
            facecolor=color,
            edgecolor=color,
            alpha=alpha,
            linewidth=linewidth,
            zorder=0,  # Draw behind nodes
        )
        ax.add_patch(circle)

    logger.debug(f"Drew {len(community_result.communities)} community circles")
