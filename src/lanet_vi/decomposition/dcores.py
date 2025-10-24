"""D-core decomposition for directed graphs.

D-cores extend k-cores to directed graphs by considering both in-degree and out-degree.
Each node is assigned a (k_in, k_out) pair indicating its core membership based on
incoming and outgoing edges separately.

This implementation is based on the algorithm from legacy/Source/graph_dcores.cpp
"""

from typing import Dict, Optional, Tuple

import networkx as nx

from lanet_vi.logging_config import get_logger
from lanet_vi.models.config import DecompositionConfig
from lanet_vi.models.graph import Component, DecompositionResult

logger = get_logger(__name__)


def compute_dcores(
    graph: nx.DiGraph,
    config: Optional[DecompositionConfig] = None,
) -> DecompositionResult:
    """
    Compute d-core decomposition for directed graphs.

    Each node receives a (k_in, k_out) core number pair based on:
    - k_in: maximum k such that node has in-degree >= k in the k-in-core
    - k_out: maximum k such that node has out-degree >= k in the k-out-core

    Parameters
    ----------
    graph : nx.DiGraph
        Directed input graph
    config : Optional[DecompositionConfig]
        Decomposition configuration

    Returns
    -------
    DecompositionResult
        D-core decomposition results with (k_in, k_out) pairs

    Raises
    ------
    ValueError
        If graph is not directed

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(0,1), (1,2), (2,0)])
    >>> result = compute_dcores(G)
    >>> result.node_indices[0]  # (k_in, k_out) for node 0
    (1, 1)

    Notes
    -----
    For undirected graphs, use compute_kcores instead.
    """
    if not graph.is_directed():
        raise ValueError("D-core decomposition requires a directed graph. Use compute_kcores for undirected graphs.")

    if config is None:
        config = DecompositionConfig()

    logger.info(f"Computing d-core decomposition for directed graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Compute in-degree and out-degree cores separately
    logger.debug("Computing in-degree cores")
    in_cores = _compute_directional_cores(graph, direction="in")

    logger.debug("Computing out-degree cores")
    out_cores = _compute_directional_cores(graph, direction="out")

    # Combine into (k_in, k_out) pairs
    node_indices = {}
    max_in = 0
    max_out = 0

    for node in graph.nodes():
        k_in = in_cores.get(node, 0)
        k_out = out_cores.get(node, 0)
        node_indices[node] = (k_in, k_out)

        max_in = max(max_in, k_in)
        max_out = max(max_out, k_out)

    # For compatibility with visualization, use max(k_in, k_out) as primary index
    # Store full (k_in, k_out) in metadata
    simple_indices = {node: max(pair) for node, pair in node_indices.items()}

    logger.info(f"D-core decomposition complete: max_in={max_in}, max_out={max_out}")

    return DecompositionResult(
        decomp_type="dcores",
        node_indices=simple_indices,  # Use max for visualization
        max_index=max(max_in, max_out),
        min_index=1,
        metadata={
            "d_cores": node_indices,  # Full (k_in, k_out) pairs
            "max_in_core": max_in,
            "max_out_core": max_out,
        },
    )


def _compute_directional_cores(
    graph: nx.DiGraph,
    direction: str = "in",
) -> Dict[int, int]:
    """
    Compute k-core based on either in-degree or out-degree.

    Parameters
    ----------
    graph : nx.DiGraph
        Directed graph
    direction : str
        Either "in" for in-degree cores or "out" for out-degree cores

    Returns
    -------
    Dict[int, int]
        Mapping from node to core number
    """
    # Create a copy to modify
    G = graph.copy()

    # Get appropriate degree function
    if direction == "in":
        degree_func = G.in_degree
    else:
        degree_func = G.out_degree

    # Initialize core numbers
    core_numbers = {node: 0 for node in G.nodes()}

    current_k = 1

    while G.number_of_nodes() > 0:
        # Find nodes with degree < current_k
        to_remove = [node for node, deg in degree_func() if deg < current_k]

        if not to_remove:
            # All remaining nodes have degree >= current_k
            # Assign them this core number and increment k
            for node in G.nodes():
                core_numbers[node] = current_k

            current_k += 1

            # Remove one layer at a time
            # Find nodes with degree exactly current_k-1
            to_remove = [node for node, deg in degree_func() if deg < current_k]

            if not to_remove:
                # All nodes have degree >= current_k
                continue

        # Remove nodes with insufficient degree
        G.remove_nodes_from(to_remove)

    return core_numbers


def find_components_by_dcore(
    graph: nx.DiGraph,
    decomposition: DecompositionResult,
) -> DecompositionResult:
    """
    Find connected components within each d-core level.

    Parameters
    ----------
    graph : nx.DiGraph
        Directed input graph
    decomposition : DecompositionResult
        D-core decomposition result

    Returns
    -------
    DecompositionResult
        Updated decomposition with component information

    Notes
    -----
    Uses weakly connected components for directed graphs.
    """
    logger.debug("Finding components for d-core decomposition")

    # Get d-core pairs from metadata
    if "d_cores" not in decomposition.metadata:
        logger.warning("No d-core pairs in metadata, using simple indices")
        d_cores = {node: (idx, idx) for node, idx in decomposition.node_indices.items()}
    else:
        d_cores = decomposition.metadata["d_cores"]

    # Group nodes by (k_in, k_out) pair
    from collections import defaultdict
    cores_dict = defaultdict(list)

    for node, (k_in, k_out) in d_cores.items():
        # Use simple max index for grouping
        k = max(k_in, k_out)
        cores_dict[k].append(node)

    # Find components in each core level
    components = []
    component_id = 0

    for core_level in sorted(cores_dict.keys(), reverse=True):
        nodes_in_core = cores_dict[core_level]

        # Create subgraph
        subgraph = graph.subgraph(nodes_in_core)

        # Find weakly connected components
        for comp_nodes in nx.weakly_connected_components(subgraph):
            component = Component(
                id=component_id,
                index=core_level,
                nodes=list(comp_nodes),
                size=len(comp_nodes),
            )
            components.append(component)
            component_id += 1

    decomposition.components = components
    logger.info(f"Found {len(components)} components in d-core decomposition")

    return decomposition
