"""D-core decomposition for directed graphs.

D-cores extend k-cores to directed graphs by considering both in-degree and out-degree.
Each node is assigned a (k_in, k_out) pair indicating its core membership based on
incoming and outgoing edges separately.

Each direction is peeled independently: ``k_in`` is the largest k such that the node
belongs to the subgraph where every node has in-degree >= k, and ``k_out`` likewise for
out-degree. ``compute_dcore_table`` gives the full (k, l)-core table of Giatsidis et al.
(for every out-degree threshold l, the largest k with the node in the (k, l)-core), the
``dcores_list.txt`` of the C++ 4.0.0 driver.
"""

import heapq
from collections import defaultdict

import networkx as nx

from lanet_vi.decomposition.kcores import _without_self_loops
from lanet_vi.logging_config import get_logger
from lanet_vi.models.config import DecompositionConfig
from lanet_vi.models.graph import Component, DecompositionResult

logger = get_logger(__name__)


def compute_dcores(
    graph: nx.DiGraph,
    config: DecompositionConfig | None = None,
) -> DecompositionResult:
    """
    Compute d-core decomposition for directed graphs.

    Each node receives a (k_in, k_out) core number pair based on:
    - k_in: maximum k such that node has in-degree >= k in the k-in-core
    - k_out: maximum k such that node has out-degree >= k in the k-out-core

    Self-loops are ignored (they count for neither degree).

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
    >>> G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    >>> result = compute_dcores(G)
    >>> result.node_indices[0]  # max(k_in, k_out), used for layout/coloring
    1
    >>> result.metadata["d_cores"][0]  # full (k_in, k_out) pair for node 0
    (1, 1)

    Notes
    -----
    For undirected graphs, use compute_kcores instead.
    """
    if not graph.is_directed():
        raise ValueError(
            "D-core decomposition requires a directed graph. "
            "Use compute_kcores for undirected graphs."
        )

    if config is None:
        config = DecompositionConfig()

    logger.info(
        f"Computing d-core decomposition for directed graph with {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges"
    )

    # Self-loops count for neither degree (the edge-list reader drops them; a node is
    # not its own neighbor), as in compute_dcore_table
    graph = _without_self_loops(graph, "the d-core decomposition")

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
        min_index=min(simple_indices.values()) if simple_indices else 0,
        metadata={
            "d_cores": node_indices,  # Full (k_in, k_out) pairs
            "max_in_core": max_in,
            "max_out_core": max_out,
        },
    )


def _compute_directional_cores(
    graph: nx.DiGraph,
    direction: str = "in",
) -> dict[int, int]:
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


def compute_dcore_table(graph: nx.DiGraph) -> dict[int, dict[int, int]]:
    """
    Compute the (k, l)-core table of a directed graph.

    The (k, l)-core (Giatsidis, Thilikos & Vazirgiannis, 2011) is the largest subgraph
    in which every node has in-degree >= k and out-degree >= l. For every l from 0 up to
    the last non-empty (0, l)-core, the table gives each node of that core the largest k
    such that the node belongs to the (k, l)-core. Row 0 is the in-core number
    (``k_in`` of ``compute_dcores``); a node absent from a row is not in the (0, l)-core.

    Parameters
    ----------
    graph : nx.DiGraph
        Directed input graph

    Returns
    -------
    Dict[int, Dict[int, int]]
        ``{l: {node: k}}``

    Raises
    ------
    ValueError
        If the graph is not directed

    Examples
    --------
    >>> G = nx.DiGraph([(0, 1), (1, 2), (2, 0), (2, 3)])
    >>> table = compute_dcore_table(G)
    >>> sorted(table[0].items())  # in-core numbers
    [(0, 1), (1, 1), (2, 1), (3, 1)]
    >>> sorted(table[1].items())  # node 3 has out-degree 0: not in the (0, 1)-core
    [(0, 1), (1, 1), (2, 1)]

    Notes
    -----
    This is what the C++ 4.0.0 driver computed for a directed graph (``-directed``): it
    wrote the table to ``dcores_list.txt`` as ``node k l`` lines and drew nothing. That
    code kept a node in the in-degree peeling after its out-degree had dropped below l,
    so it reported a larger k than the definition for such nodes; this implementation
    follows the definition (a node leaves the (k, l)-core as soon as either degree
    fails), which is checked against a brute-force one in the tests.
    """
    if not graph.is_directed():
        raise ValueError("The (k, l)-core table requires a directed graph")

    table: dict[int, dict[int, int]] = {}
    # Nodes of the (0, l)-core; the cores are nested, so the set only shrinks with l
    alive = set(graph.nodes())
    out_degree = {v: sum(1 for w in graph.successors(v) if w != v) for v in alive}
    in_degree = {v: sum(1 for u in graph.predecessors(v) if u != v) for v in alive}
    out_min = 0
    while True:
        # Prune to the (0, l)-core: drop every node whose out-degree fell below l
        pending = [v for v in alive if out_degree[v] < out_min]
        while pending:
            v = pending.pop()
            if v not in alive:
                continue
            alive.remove(v)
            for u in graph.predecessors(v):
                if u in alive and u != v:
                    out_degree[u] -= 1
                    if out_degree[u] < out_min:
                        pending.append(u)
            for w in graph.successors(v):
                if w in alive and w != v:
                    in_degree[w] -= 1
        if not alive:
            break
        table[out_min] = _peel_in_degree(graph, alive, dict(in_degree), dict(out_degree), out_min)
        out_min += 1
    return table


def _peel_in_degree(
    graph: nx.DiGraph,
    core: set[int],
    in_degree: dict[int, int],
    out_degree: dict[int, int],
    out_min: int,
) -> dict[int, int]:
    """Largest k with each node of ``core`` (the (0, l)-core) in the (k, l)-core.

    Batagelj-Zaversnik peeling by in-degree, where a node whose out-degree drops below
    ``out_min`` (the l) is removed at once with the current k (it is in the (k, l)-core,
    not in the (k + 1, l)-core). The degree dictionaries are the degrees within ``core``.
    """
    remaining = set(core)
    heap = [(in_degree[v], v) for v in remaining]
    heapq.heapify(heap)
    forced: list[int] = []
    result: dict[int, int] = {}
    k = 0
    while remaining:
        if forced:
            v = forced.pop()
            if v not in remaining:
                continue
        else:
            degree, v = heapq.heappop(heap)
            if v not in remaining or degree != in_degree[v]:
                continue  # stale heap entry
            k = max(k, degree)
        result[v] = k
        remaining.remove(v)
        for w in graph.successors(v):
            if w in remaining and w != v:
                in_degree[w] -= 1
                heapq.heappush(heap, (in_degree[w], w))
        for u in graph.predecessors(v):
            if u in remaining and u != v:
                out_degree[u] -= 1
                if out_degree[u] < out_min:
                    forced.append(u)
    return result


def find_components_by_dcore(
    graph: nx.DiGraph,
    decomposition: DecompositionResult,
) -> DecompositionResult:
    """
    Find weakly connected components within each d-core level.

    Nodes are grouped by ``max(k_in, k_out)``, the same value stored in
    ``decomposition.node_indices`` and used by the layout, so the resulting
    components line up with the rings drawn for the other decompositions.

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
    """
    logger.debug("Finding components for d-core decomposition")

    nodes_by_level: dict[int, list[int]] = defaultdict(list)
    for node, level in decomposition.node_indices.items():
        nodes_by_level[level].append(node)

    components: list[Component] = []
    component_id = 0

    for core_level in sorted(nodes_by_level, reverse=True):
        subgraph = graph.subgraph(nodes_by_level[core_level])

        for comp_nodes in nx.weakly_connected_components(subgraph):
            components.append(
                Component(
                    component_id=component_id,
                    nodes=list(comp_nodes),
                    shell_index=core_level,
                    size=len(comp_nodes),
                )
            )
            component_id += 1

    decomposition.components = components
    logger.info(f"Found {len(components)} components in d-core decomposition")

    return decomposition
