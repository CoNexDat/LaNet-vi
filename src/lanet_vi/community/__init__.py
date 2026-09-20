"""Community detection algorithms for network analysis.

This module provides community detection functionality including:
- Louvain algorithm for modularity optimization
- Greedy modularity maximization
- ``detect_communities``, which runs the algorithm named in a ``CommunityConfig``

The overlays that draw a ``CommunityResult`` on a picture live in
``lanet_vi.visualization.community_viz``.
"""

import networkx as nx

from lanet_vi.community.base import Community, CommunityResult
from lanet_vi.community.louvain import (
    detect_communities_greedy_modularity,
    detect_communities_louvain,
)
from lanet_vi.models.config import CommunityConfig

__all__ = [
    "Community",
    "CommunityResult",
    "detect_communities",
    "detect_communities_greedy_modularity",
    "detect_communities_louvain",
]


def detect_communities(
    graph: nx.Graph,
    config: CommunityConfig,
    seed: int | None = None,
) -> CommunityResult:
    """Run the community detection algorithm named in ``config``.

    Parameters
    ----------
    graph : nx.Graph
        Input graph; directed graphs are analyzed as their undirected version
    config : CommunityConfig
        ``algorithm`` and ``resolution`` are read; ``detect_communities`` is not (the
        caller decided to detect)
    seed : Optional[int]
        Random seed of the Louvain algorithm (greedy modularity is deterministic)

    Returns
    -------
    CommunityResult
        Detected communities with modularity score

    Raises
    ------
    ValueError
        If ``config.algorithm`` is not a known algorithm.
    """
    if config.algorithm == "louvain":
        return detect_communities_louvain(graph, resolution=config.resolution, seed=seed)
    if config.algorithm == "greedy_modularity":
        return detect_communities_greedy_modularity(graph, resolution=config.resolution)
    raise ValueError(f"Unknown community algorithm: {config.algorithm!r}")
