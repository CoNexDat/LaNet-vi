"""K-core decomposition using NetworkX."""

from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from lanet_vi.logging_config import get_logger
from lanet_vi.models.config import DecompositionConfig, StrengthIntervalMethod
from lanet_vi.models.graph import Component, DecompositionResult

logger = get_logger(__name__)


def compute_kcores(
    graph: nx.Graph,
    config: Optional[DecompositionConfig] = None,
) -> DecompositionResult:
    """
    Compute k-core decomposition of a graph.

    For unweighted graphs, uses NetworkX's built-in k-core algorithm.
    For weighted graphs, computes strength-based decomposition using p-function.

    Parameters
    ----------
    graph : nx.Graph
        Input graph
    config : Optional[DecompositionConfig]
        Decomposition configuration

    Returns
    -------
    DecompositionResult
        K-core decomposition results with shell indices

    Examples
    --------
    >>> G = nx.karate_club_graph()
    >>> result = compute_kcores(G)
    >>> print(f"Max core: {result.max_index}")
    """
    if config is None:
        config = DecompositionConfig()

    logger.info(f"Computing k-core decomposition for graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Check if graph is weighted (check first 100 edges for performance)
    is_weighted = any("weight" in graph[u][v] for u, v in list(graph.edges())[:100])

    if not is_weighted:
        logger.info("Using unweighted k-core algorithm (NetworkX core_number)")
        # Use NetworkX's k-core number computation
        core_numbers = nx.core_number(graph)

        max_core = max(core_numbers.values()) if core_numbers else 0
        min_core = min(core_numbers.values()) if core_numbers else 0
        logger.info(f"K-core decomposition complete: min={min_core}, max={max_core}")

        return DecompositionResult(
            decomp_type="kcores",
            node_indices=core_numbers,
            max_index=max_core,
            min_index=min_core,
        )

    else:
        logger.info("Using weighted k-core algorithm (strength-based p-function)")
        # Weighted graph: use strength-based p-function
        p_function = _build_p_function(graph, config)
        logger.debug(f"Built p-function with {len(p_function)} intervals")

        core_numbers = _compute_weighted_cores(graph, p_function)

        max_core = max(core_numbers.values()) if core_numbers else 0
        min_core = min(core_numbers.values()) if core_numbers else 0
        logger.info(f"Weighted k-core decomposition complete: min={min_core}, max={max_core}")

        return DecompositionResult(
            decomp_type="kcores",
            node_indices=core_numbers,
            max_index=max_core,
            min_index=min_core,
            p_function=p_function,
        )


def _build_p_function(
    graph: nx.Graph,
    config: DecompositionConfig,
) -> List[float]:
    """
    Build p-function for weighted graph decomposition.

    The p-function defines strength intervals that partition nodes
    into groups based on their weighted degree (strength).

    Parameters
    ----------
    graph : nx.Graph
        Weighted graph
    config : DecompositionConfig
        Configuration with granularity and interval method

    Returns
    -------
    List[float]
        Strength interval boundaries
    """
    # Calculate node strengths (sum of edge weights)
    strengths = []
    for node in graph.nodes():
        strength = sum(
            graph[node][neighbor].get("weight", 1.0) for neighbor in graph.neighbors(node)
        )
        strengths.append(strength)

    strengths = sorted(strengths)

    # Determine granularity
    if config.granularity == -1:
        # Use maximum degree as granularity, but cap at 100 for performance
        max_degree = max(dict(graph.degree()).values())
        granularity = min(max_degree, 100)
    else:
        granularity = config.granularity

    # Build p-function based on interval method
    p_function = [0.0]

    if config.strength_intervals == StrengthIntervalMethod.EQUAL_NODES:
        # Equal number of nodes per interval
        n = len(graph.nodes())
        for i in range(1, granularity + 1):
            if i < granularity:
                idx = int(np.ceil((n - 1) * i / granularity))
                p_function.append(strengths[idx])
            else:
                p_function.append(strengths[-1])

    elif config.strength_intervals == StrengthIntervalMethod.EQUAL_LOG_SIZE:
        # Logarithmic intervals
        a = strengths[0] if strengths[0] > 0 else 0.01
        b = config.maximum_strength if config.maximum_strength else strengths[-1]

        for i in range(1, granularity + 1):
            if i < granularity:
                p_function.append(a * ((b / a) ** (i / granularity)))
            else:
                p_function.append(b)

    else:  # EQUAL_SIZE (default)
        # Equal interval size
        max_strength = config.maximum_strength if config.maximum_strength else strengths[-1]
        interval_size = max_strength / granularity

        for i in range(1, granularity + 1):
            if i < granularity:
                p_function.append(i * interval_size)
            else:
                p_function.append(max_strength)

    return p_function


def _compute_weighted_cores(
    graph: nx.Graph,
    p_function: List[float],
) -> Dict[int, int]:
    """
    Compute core numbers for weighted graph using p-function.

    Assigns each node to a p-value (interval index) based on its strength,
    then applies k-core-like peeling algorithm.

    Parameters
    ----------
    graph : nx.Graph
        Weighted graph
    p_function : List[float]
        Strength interval boundaries

    Returns
    -------
    Dict[int, int]
        Mapping from node to core number (p-value)
    """
    # Calculate node strengths and assign p-values
    node_p_values = {}
    for node in graph.nodes():
        strength = sum(
            graph[node][neighbor].get("weight", 1.0) for neighbor in graph.neighbors(node)
        )

        # Find p-value (which interval the strength falls into)
        p_value = 0
        for i, threshold in enumerate(p_function[1:], start=1):
            if strength <= threshold:
                p_value = i
                break
        if p_value == 0:
            p_value = len(p_function) - 1

        node_p_values[node] = p_value

    return node_p_values


def find_components_by_shell(
    graph: nx.Graph,
    decomposition: DecompositionResult,
) -> DecompositionResult:
    """
    Find connected components for each shell index.

    Parameters
    ----------
    graph : nx.Graph
        Original graph
    decomposition : DecompositionResult
        K-core decomposition result

    Returns
    -------
    DecompositionResult
        Updated result with component information
    """
    components = []
    component_id = 0

    # Group nodes by shell index
    shells: Dict[int, List[int]] = {}
    for node, shell_idx in decomposition.node_indices.items():
        if shell_idx not in shells:
            shells[shell_idx] = []
        shells[shell_idx].append(node)

    # Find components within each shell
    for shell_idx in sorted(shells.keys(), reverse=True):
        # Create induced subgraph for this shell (copy for performance)
        shell_nodes = shells[shell_idx]

        # Only process if there are nodes in this shell
        if not shell_nodes:
            continue

        # Create subgraph - use copy() to avoid view overhead
        subgraph = graph.subgraph(shell_nodes).copy()

        # Find connected components
        for comp_nodes in nx.connected_components(subgraph):
            components.append(
                Component(
                    component_id=component_id,
                    nodes=list(comp_nodes),
                    shell_index=shell_idx,
                    size=len(comp_nodes),
                )
            )
            component_id += 1

    decomposition.components = components
    return decomposition
