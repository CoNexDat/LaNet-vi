"""Community visualization utilities.

This module provides functions for visualizing network communities, including:
- Coloring nodes by community membership
- Drawing community boundaries
- Creating community-based color palettes
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

from lanet_vi.community.base import CommunityResult
from lanet_vi.logging_config import get_logger

logger = get_logger(__name__)


def get_community_colors(
    num_communities: int,
    colormap: str = "tab20",
) -> List[Tuple[float, float, float]]:
    """Generate distinct colors for communities.

    Parameters
    ----------
    num_communities : int
        Number of communities to generate colors for
    colormap : str
        Matplotlib colormap name (default: "tab20" for up to 20 distinct colors)

    Returns
    -------
    List[Tuple[float, float, float]]
        List of RGB color tuples

    Notes
    -----
    For more than 20 communities, consider using "hsv" or "rainbow" colormaps.
    """
    if num_communities <= 20:
        cmap = plt.cm.get_cmap("tab20")
    elif num_communities <= 40:
        # Combine tab20 with tab20b/tab20c
        cmap = plt.cm.get_cmap("tab20b")
    else:
        # Use continuous colormap for many communities
        cmap = plt.cm.get_cmap("hsv")

    colors = []
    for i in range(num_communities):
        colors.append(cmap(i / max(num_communities, 1)))

    return colors


def assign_node_colors_by_community(
    community_result: CommunityResult,
    node_positions: Dict[int, Tuple[float, float]],
    colormap: str = "tab20",
) -> Dict[int, Tuple[float, float, float]]:
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

    # Get color palette
    community_colors = get_community_colors(
        community_result.num_communities,
        colormap=colormap,
    )

    # Assign colors to nodes
    node_colors = {}
    for node in node_positions:
        comm_id = community_result.get_node_community(node)
        if comm_id is not None:
            node_colors[node] = community_colors[comm_id][:3]  # RGB only
        else:
            # Default gray for nodes not in any community
            node_colors[node] = (0.7, 0.7, 0.7)

    return node_colors


def draw_community_boundaries(
    ax: plt.Axes,
    community_result: CommunityResult,
    node_positions: Dict[int, Tuple[float, float]],
    alpha: float = 0.2,
    linewidth: float = 2.0,
    colormap: str = "tab20",
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

    Notes
    -----
    Only draws boundaries for communities with 3 or more nodes (required for convex hull).
    """
    logger.debug(f"Drawing boundaries for {community_result.num_communities} communities")

    # Get color palette
    community_colors = get_community_colors(
        community_result.num_communities,
        colormap=colormap,
    )

    patches = []
    colors = []

    for community in community_result.communities:
        # Get positions of nodes in this community
        points = []
        for node in community.nodes:
            if node in node_positions:
                points.append(node_positions[node])

        # Need at least 3 points for convex hull
        if len(points) < 3:
            continue

        # Compute convex hull
        try:
            points_array = np.array(points)
            hull = ConvexHull(points_array)

            # Create polygon from hull vertices
            hull_points = points_array[hull.vertices]
            polygon = Polygon(hull_points, closed=True)

            patches.append(polygon)
            colors.append(community_colors[community.id])

        except Exception as e:
            logger.warning(
                f"Could not compute convex hull for community {community.id}: {e}"
            )
            continue

    # Draw all patches
    if patches:
        collection = PatchCollection(
            patches,
            facecolors=colors,
            alpha=alpha,
            edgecolors=colors,
            linewidths=linewidth,
        )
        ax.add_collection(collection)
        logger.debug(f"Drew {len(patches)} community boundaries")


def draw_community_circles(
    ax: plt.Axes,
    community_result: CommunityResult,
    node_positions: Dict[int, Tuple[float, float]],
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

    # Get color palette
    community_colors = get_community_colors(
        community_result.num_communities,
        colormap=colormap,
    )

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
        max_dist = np.max(np.linalg.norm(points_array - center, axis=1))
        radius = max_dist * (1.0 + padding)

        # Draw circle
        color = community_colors[community.id]
        circle = plt.Circle(
            center,
            radius,
            color=color,
            alpha=alpha,
            fill=True,
            linewidth=linewidth,
            edgecolor=color,
            zorder=0,  # Draw behind nodes
        )
        ax.add_patch(circle)

    logger.debug(f"Drew {len(community_result.communities)} community circles")
