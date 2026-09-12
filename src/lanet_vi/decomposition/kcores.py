"""K-core decomposition using NetworkX."""

import networkx as nx
import numpy as np

from lanet_vi.logging_config import get_logger
from lanet_vi.models.config import DecompositionConfig, StrengthIntervalMethod
from lanet_vi.models.graph import Component, DecompositionResult

logger = get_logger(__name__)


def compute_kcores(
    graph: nx.Graph,
    config: DecompositionConfig | None = None,
    weighted: bool | None = None,
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
    weighted : Optional[bool]
        Force the weighted (strength-based) or unweighted algorithm. ``None``
        (default) picks the weighted one if any edge has a ``weight`` attribute.

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

    logger.info(
        f"Computing k-core decomposition for graph with {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges"
    )

    graph = _without_self_loops(graph)

    # Weighted if the caller says so; otherwise if any edge carries a weight
    # (short-circuits at the first weighted edge)
    is_weighted = (
        weighted
        if weighted is not None
        else any("weight" in data for _, _, data in graph.edges(data=True))
    )

    graph = _merge_parallel_edges(graph) if is_weighted and graph.is_multigraph() else graph

    if not is_weighted:
        logger.info("Using unweighted k-core algorithm (NetworkX core_number)")
        # Use NetworkX's k-core number computation
        core_numbers = _core_number(graph)

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


def _without_self_loops(graph: nx.Graph) -> nx.Graph:
    """Return ``graph`` without self-loops (copied only if any exist).

    The C++ LaNet-vi ignored the self-loop contribution to the core index and
    NetworkX's ``core_number`` refuses self-loops altogether.
    """
    n_loops = nx.number_of_selfloops(graph)
    if not n_loops:
        return graph
    logger.warning(f"Ignoring {n_loops} self-loop(s) for the k-core decomposition")
    graph = graph.copy()
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def _merge_parallel_edges(graph: nx.Graph) -> nx.Graph:
    """Collapse a weighted multigraph into a simple graph, summing parallel weights.

    The C++ ``-multigraph -weighted`` mode summed the weights of parallel edges
    into the node strength, which is what the p-function needs.
    """
    logger.info("Merging parallel edges: strength is the sum of their weights")
    simple: nx.Graph = nx.DiGraph() if graph.is_directed() else nx.Graph()
    simple.add_nodes_from(graph.nodes(data=True))
    for u, v, data in graph.edges(data=True):
        w = float(data.get("weight", 1.0))
        if simple.has_edge(u, v):
            simple[u][v]["weight"] += w
        else:
            simple.add_edge(u, v, weight=w)
    return simple


def _core_number(graph: nx.Graph) -> dict[int, int]:
    """Core numbers where parallel edges count towards the degree.

    Delegates to ``nx.core_number`` for simple graphs and runs the same
    Batagelj-Zaversnik peeling on multigraphs, as the C++ ``-multigraph``
    mode did.
    """
    if not graph.is_multigraph():
        return dict(nx.core_number(graph))

    degrees = dict(graph.degree())  # in + out for directed graphs, multiplicity counted
    core = dict(degrees)
    directed = graph.is_directed()

    def incident(node: int) -> dict[int, int]:
        """Neighbours of ``node`` with the number of edges shared in either direction."""
        counts: dict[int, int] = {}
        for nb in graph.successors(node) if directed else graph.neighbors(node):
            counts[nb] = counts.get(nb, 0) + graph.number_of_edges(node, nb)
        if directed:
            for nb in graph.predecessors(node):
                counts[nb] = counts.get(nb, 0) + graph.number_of_edges(nb, node)
        return counts

    max_degree = max(degrees.values(), default=0)
    bins: list[list[int]] = [[] for _ in range(max_degree + 1)]
    for node, degree in degrees.items():
        bins[degree].append(node)

    removed: set[int] = set()
    for k in range(max_degree + 1):
        bucket = bins[k]
        while bucket:
            node = bucket.pop()
            if node in removed:
                continue
            removed.add(node)
            core[node] = k
            for neighbour, multiplicity in incident(node).items():
                if neighbour in removed or core[neighbour] <= k:
                    continue
                core[neighbour] = max(k, core[neighbour] - multiplicity)
                bins[core[neighbour]].append(neighbour)
    return core


def _build_p_function(
    graph: nx.Graph,
    config: DecompositionConfig,
) -> list[float]:
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
    p_function: list[float],
) -> dict[int, int]:
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
    shells: dict[int, list[int]] = {}
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
