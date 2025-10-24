"""Louvain community detection algorithm.

This module implements community detection using the Louvain method for modularity
optimization. The implementation uses NetworkX's community detection functions.
"""

from typing import Optional

import networkx as nx
from networkx.algorithms import community as nx_community

from lanet_vi.community.base import Community, CommunityResult
from lanet_vi.logging_config import get_logger

logger = get_logger(__name__)


def detect_communities_louvain(
    graph: nx.Graph,
    weight: Optional[str] = "weight",
    resolution: float = 1.0,
    seed: Optional[int] = None,
) -> CommunityResult:
    """Detect communities using the Louvain algorithm.

    The Louvain method is a greedy optimization algorithm that attempts to maximize
    the modularity of a partition of the network. It works on both weighted and
    unweighted graphs.

    Parameters
    ----------
    graph : nx.Graph
        Input graph (undirected). For directed graphs, the graph will be converted
        to undirected first.
    weight : Optional[str]
        Edge attribute to use as weight (default: "weight"). Set to None for
        unweighted graphs.
    resolution : float
        Resolution parameter for modularity (default: 1.0). Higher values lead
        to more communities.
    seed : Optional[int]
        Random seed for reproducibility (default: None)

    Returns
    -------
    CommunityResult
        Detected communities with modularity score

    Examples
    --------
    >>> G = nx.karate_club_graph()
    >>> result = detect_communities_louvain(G)
    >>> result.num_communities
    4
    >>> result.modularity > 0.4
    True

    Notes
    -----
    This function uses NetworkX's `community.louvain_communities` which implements
    the algorithm from:
    Blondel, V.D. et al. "Fast unfolding of communities in large networks."
    Journal of Statistical Mechanics (2008).
    """
    # Convert directed to undirected if needed
    if graph.is_directed():
        logger.info("Converting directed graph to undirected for community detection")
        graph = graph.to_undirected()

    logger.info(
        f"Running Louvain community detection on graph with {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges (resolution={resolution})"
    )

    # Check if graph has weights
    has_weights = weight is not None and any(
        weight in data for _, _, data in graph.edges(data=True)
    )

    if not has_weights and weight is not None:
        logger.debug("No edge weights found, using unweighted Louvain")
        weight = None

    # Run Louvain algorithm
    communities_sets = nx_community.louvain_communities(
        graph,
        weight=weight,
        resolution=resolution,
        seed=seed,
    )

    # Convert to Community objects
    communities = []
    node_to_community = {}

    for comm_id, node_set in enumerate(communities_sets):
        nodes_list = sorted(list(node_set))
        community = Community(
            id=comm_id,
            nodes=nodes_list,
            size=len(nodes_list),
        )
        communities.append(community)

        # Build reverse mapping
        for node in nodes_list:
            node_to_community[node] = comm_id

    # Calculate modularity
    modularity = nx_community.modularity(graph, communities_sets, weight=weight)

    logger.info(
        f"Louvain detection complete: {len(communities)} communities, "
        f"modularity={modularity:.4f}"
    )

    return CommunityResult(
        algorithm="louvain",
        communities=communities,
        node_to_community=node_to_community,
        num_communities=len(communities),
        modularity=modularity,
    )


def detect_communities_greedy_modularity(
    graph: nx.Graph,
    weight: Optional[str] = "weight",
) -> CommunityResult:
    """Detect communities using greedy modularity maximization.

    This is an alternative to Louvain that may be faster on some graphs but
    typically produces lower modularity scores.

    Parameters
    ----------
    graph : nx.Graph
        Input graph (undirected)
    weight : Optional[str]
        Edge attribute to use as weight (default: "weight")

    Returns
    -------
    CommunityResult
        Detected communities with modularity score

    Notes
    -----
    Uses NetworkX's `community.greedy_modularity_communities`.
    """
    if graph.is_directed():
        logger.info("Converting directed graph to undirected for community detection")
        graph = graph.to_undirected()

    logger.info(
        f"Running greedy modularity community detection on graph with "
        f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
    )

    # Run greedy modularity algorithm
    communities_sets = nx_community.greedy_modularity_communities(
        graph,
        weight=weight,
    )

    # Convert to Community objects
    communities = []
    node_to_community = {}

    for comm_id, node_set in enumerate(communities_sets):
        nodes_list = sorted(list(node_set))
        community = Community(
            id=comm_id,
            nodes=nodes_list,
            size=len(nodes_list),
        )
        communities.append(community)

        for node in nodes_list:
            node_to_community[node] = comm_id

    # Calculate modularity
    modularity = nx_community.modularity(graph, communities_sets, weight=weight)

    logger.info(
        f"Greedy modularity detection complete: {len(communities)} communities, "
        f"modularity={modularity:.4f}"
    )

    return CommunityResult(
        algorithm="greedy_modularity",
        communities=communities,
        node_to_community=node_to_community,
        num_communities=len(communities),
        modularity=modularity,
    )
