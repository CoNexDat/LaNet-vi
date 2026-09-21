"""K-core decomposition (degree-based and, for weighted graphs, strength-based)."""

import math
from bisect import bisect_left
from pathlib import Path

import networkx as nx
import numpy as np

from lanet_vi.decomposition.components import components_by_index
from lanet_vi.logging_config import get_logger
from lanet_vi.models.config import DecompositionConfig, StrengthIntervalMethod
from lanet_vi.models.graph import DecompositionResult

logger = get_logger(__name__)


def compute_kcores(
    graph: nx.Graph,
    config: DecompositionConfig | None = None,
    weighted: bool | None = None,
    p_function: list[float] | None = None,
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
    p_function : Optional[list[float]]
        Strength interval boundaries to reuse instead of building them from this
        graph (the C++ ``findCores(&pf)``, used when re-decomposing the subgraph of
        ``from_layer``). Ignored for unweighted graphs.

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
    # (short-circuits at the first weighted edge). An edgeless graph has no
    # strengths to bin, so it always takes the unweighted path (all cores 0).
    is_weighted = graph.number_of_edges() > 0 and (
        weighted
        if weighted is not None
        else any("weight" in data for _, _, data in graph.edges(data=True))
    )

    max_degree = max((d for _, d in graph.degree()), default=0)  # parallel edges count
    if is_weighted:
        _check_weights_non_negative(graph)
        if graph.is_multigraph() or graph.is_directed():
            graph = _as_weighted_simple_graph(graph)

    if not is_weighted:
        logger.info("Using unweighted k-core algorithm")
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
        # Weighted graph: bin strengths with the p-function, then peel
        strengths = _node_strengths(graph)
        if p_function is None:
            p_function = _build_p_function(strengths, max_degree, config)
            logger.debug(f"Built p-function with {len(p_function)} intervals")
        else:
            logger.debug(f"Reusing a p-function with {len(p_function)} intervals")

        core_numbers = _compute_weighted_cores(graph, strengths, p_function)

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


def _without_self_loops(graph: nx.Graph, what: str = "the k-core decomposition") -> nx.Graph:
    """Return ``graph`` without self-loops (copied only if any exist).

    The C++ LaNet-vi ignored the self-loop contribution to the core index and
    NetworkX's ``core_number`` refuses self-loops altogether.
    """
    n_loops = nx.number_of_selfloops(graph)
    if not n_loops:
        return graph
    logger.warning(f"Ignoring {n_loops} self-loop(s) for {what}")
    graph = graph.copy()
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def _as_weighted_simple_graph(graph: nx.Graph) -> nx.Graph:
    """Collapse a weighted multigraph or digraph into an undirected simple graph.

    Weights of parallel and reciprocal edges are summed, so a node's strength is
    the total weight incident to it in either direction. This matches the C++
    ``-multigraph -weighted`` mode (parallel weights summed into the strength)
    and keeps the weighted path consistent with the unweighted one, which uses
    in-degree plus out-degree on directed graphs.
    """
    logger.info("Merging parallel/reciprocal edges: strength is the sum of their weights")
    simple = nx.Graph()
    simple.add_nodes_from(graph.nodes(data=True))
    for u, v, data in graph.edges(data=True):
        w = float(data.get("weight", 1.0))
        if simple.has_edge(u, v):
            simple[u][v]["weight"] += w
        else:
            simple.add_edge(u, v, weight=w)
    return simple


def _core_number(graph: nx.Graph) -> dict[int, int]:
    """Core numbers by bucket peeling (Batagelj & Zaversnik), parallel edges counted.

    Every edge counts towards the degree of both endpoints: parallel edges of a
    multigraph, as the C++ ``-multigraph`` mode did, and both arcs of a mutual pair in a
    directed graph, as ``nx.core_number`` does. Gives the same numbers as
    ``nx.core_number`` on simple graphs in a fraction of its time on large ones (it
    deletes neighbors from lists).
    """
    adjacency: dict[int, dict[int, int]] = {node: {} for node in graph.nodes()}
    for u, v in graph.edges():
        adjacency[u][v] = adjacency[u].get(v, 0) + 1
        adjacency[v][u] = adjacency[v].get(u, 0) + 1
    core = {node: sum(counts.values()) for node, counts in adjacency.items()}

    max_degree = max(core.values(), default=0)
    bins: list[list[int]] = [[] for _ in range(max_degree + 1)]
    for node, degree in core.items():
        bins[degree].append(node)

    # Peel the smallest degree first; a node lowered to a new bucket is appended there
    # and its stale entries are skipped when popped
    removed: set[int] = set()
    for k in range(max_degree + 1):
        bucket = bins[k]
        while bucket:
            node = bucket.pop()
            if node in removed:
                continue
            removed.add(node)
            core[node] = k
            for neighbor, multiplicity in adjacency[node].items():
                if neighbor in removed or core[neighbor] <= k:
                    continue
                core[neighbor] = max(k, core[neighbor] - multiplicity)
                bins[core[neighbor]].append(neighbor)
    return core


def _check_weights_non_negative(graph: nx.Graph) -> None:
    """Refuse negative or non-finite weights on the raw edges, before any merge.

    The strength scale starts at 0 and the peeling relies on strengths only decreasing
    as neighbors are removed.
    """
    for u, v, data in graph.edges(data=True):
        w = data.get("weight", 1.0)
        if not math.isfinite(w) or w < 0:
            raise ValueError(
                f"Edge ({u}, {v}) has weight {w}; strength-based k-cores need finite, "
                "non-negative weights"
            )


def _node_strengths(graph: nx.Graph) -> dict[int, float]:
    """Strength (sum of incident edge weights, missing weight = 1.0) of every node."""
    strengths = {
        node: sum(data.get("weight", 1.0) for data in graph[node].values())
        for node in graph.nodes()
    }
    overflowed = [node for node, s in strengths.items() if not math.isfinite(s)]
    if overflowed:
        raise ValueError(f"Strength of node {overflowed[0]} overflows to infinity")
    return strengths


def _build_p_function(
    strengths: dict[int, float],
    max_degree: int,
    config: DecompositionConfig,
) -> list[float]:
    """
    Build the p-function (strength interval boundaries) for weighted decomposition.

    Port of ``Graph_KCores::buildPFunction``. ``p_function[0]`` is ``0.0`` and a node
    of strength ``s`` gets index ``i`` when ``p_function[i - 1] < s <= p_function[i]``
    (see ``_p_index``), so ``granularity`` intervals give indices ``1..granularity``
    and an isolated node gets ``0``, like the unweighted core number.

    Parameters
    ----------
    strengths : Dict[int, float]
        Node strengths
    max_degree : int
        Maximum degree of the input graph (parallel edges counted): the default
        granularity, as in the C++
    config : DecompositionConfig
        Granularity, interval method, maximum strength and custom intervals file

    Returns
    -------
    List[float]
        Non-decreasing boundaries starting at ``0.0``

    Notes
    -----
    The C++ 3.0.1 pushed ``0.0`` twice for ``equalNodesPerInterval`` and
    ``equalIntervalSize`` (but not for ``equalLogIntervalSize``), so its indices ran
    ``2..granularity + 1`` in those two modes. This port numbers every mode
    ``1..granularity``.
    """
    sorted_strengths = sorted(strengths.values())
    n = len(sorted_strengths)

    if config.strength_intervals == StrengthIntervalMethod.CUSTOM:
        return read_custom_intervals(config.strength_intervals_file)

    granularity = max_degree if config.granularity == -1 else config.granularity
    granularity = max(granularity, 1)  # the weighted path is only taken with edges

    p_function = [0.0]

    if config.strength_intervals == StrengthIntervalMethod.EQUAL_NODES:
        # Boundaries taken from the sorted strengths, the same number of nodes per interval
        for i in range(1, granularity + 1):
            if i < granularity:
                idx = int(np.ceil((n - 1) * i / granularity))
                p_function.append(sorted_strengths[idx])
            else:
                p_function.append(sorted_strengths[-1])

    elif config.strength_intervals == StrengthIntervalMethod.EQUAL_LOG_SIZE:
        # Geometric progression from the smallest strength to the largest. The C++
        # divided by the smallest strength as is; a zero (isolated node) would make
        # the ratio infinite, so the smallest positive strength is used instead.
        positive = [x for x in sorted_strengths if x > 0]
        b = config.maximum_strength if config.maximum_strength else sorted_strengths[-1]
        # A maximum below the smallest strength would make the progression descend
        a = min(positive[0] if positive else 1.0, b)

        for i in range(1, granularity + 1):
            if i < granularity and a > 0:
                # log-space so an extreme b/a ratio cannot overflow to inf
                log_boundary = math.log(a) + (i / granularity) * (math.log(b) - math.log(a))
                p_function.append(min(math.exp(log_boundary), b))
            else:
                p_function.append(b)  # all weights zero: every boundary is 0.0

    else:  # EQUAL_SIZE (default)
        max_strength = config.maximum_strength if config.maximum_strength else sorted_strengths[-1]
        interval_size = max_strength / granularity

        for i in range(1, granularity + 1):
            if i < granularity:
                p_function.append(i * interval_size)
            else:
                p_function.append(max_strength)

    return p_function


def read_custom_intervals(path: Path | None) -> list[float]:
    """
    Read the p-function boundaries of ``strength_intervals = custom`` from a file.

    One boundary per line, as the C++ ``-strengthsIntervalsFile``. A positive strength
    in ``(0, p1]`` gets index 1, ``(p(i-1), pi]`` index ``i``, and anything above ``pn``
    the last index ``n``; a strength of exactly 0 (isolated node) gets index 0, like
    every other interval method.
    """
    if path is None:
        raise ValueError("strength_intervals 'custom' needs strength_intervals_file")
    boundaries: list[float] = []
    with open(path) as f:
        for line in f:
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            boundaries.append(float(token))
    if not boundaries:
        raise ValueError(f"No strength intervals found in {path}")
    if any(not math.isfinite(b) or b < 0 for b in boundaries):
        raise ValueError(f"Strength intervals in {path} must be finite and non-negative")
    if boundaries != sorted(boundaries):
        raise ValueError(f"Strength intervals in {path} must be non-decreasing")
    return [0.0, *boundaries]


def _p_index(p_function: list[float], strength: float) -> int:
    """Interval index of ``strength``: smallest ``i`` with ``p_function[i] >= strength``.

    Port of ``Vertex::applyPFunction``; strengths above the last boundary get the last
    index.
    """
    return min(bisect_left(p_function, strength), len(p_function) - 1)


def _compute_weighted_cores(
    graph: nx.Graph,
    strengths: dict[int, float],
    p_function: list[float],
) -> dict[int, int]:
    """
    Strength-based k-core decomposition (port of the weighted ``findCores``).

    Every node starts at the interval index of its total strength. Shells are then
    peeled in increasing order: when a node of the current shell ``k`` is removed,
    each neighbor still above ``k`` is re-binned using only the strength it receives
    from neighbors not yet removed, and moved down to ``max(new index, k)``. The
    result is the generalized k-core: a node has index ``>= k`` iff it belongs to a
    subgraph where every node receives strength in an interval ``>= k`` from the
    others.

    The C++ re-summed a neighbor's remaining strength on every re-binning (quadratic
    in the degree of a hub); here the remaining strength is kept incrementally, so each
    edge is subtracted once. Both peel to the same fixed point.

    Parameters
    ----------
    graph : nx.Graph
        Weighted graph
    strengths : Dict[int, float]
        Total strength of every node
    p_function : List[float]
        Strength interval boundaries

    Returns
    -------
    Dict[int, int]
        Mapping from node to shell index
    """
    core = {node: _p_index(p_function, s) for node, s in strengths.items()}
    if not core:
        return core

    buckets: list[set[int]] = [set() for _ in range(len(p_function))]
    for node, k in core.items():
        buckets[k].add(node)

    remaining = dict(strengths)  # strength from neighbors not yet removed
    done: set[int] = set()
    for k, bucket in enumerate(buckets):
        while bucket:
            node = bucket.pop()
            done.add(node)
            for neighbor, data in graph[node].items():
                if neighbor in done:
                    continue
                remaining[neighbor] -= data.get("weight", 1.0)
                if core[neighbor] <= k:
                    continue
                new_k = max(_p_index(p_function, remaining[neighbor]), k)
                if new_k != core[neighbor]:
                    buckets[core[neighbor]].discard(neighbor)
                    core[neighbor] = new_k
                    buckets[new_k].add(neighbor)
    return core


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
    decomposition.components = components_by_index(graph, decomposition.node_indices)
    return decomposition
