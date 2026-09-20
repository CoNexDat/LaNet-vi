"""K-connectivity of the k-core clusters (the C++ ``-kconn``).

Port of ``computeKConnectivityWide`` / ``computeKConnectivityStrict`` and their
``conditions`` from ``graph_kcores_components.cpp``, the method of Beiró, Alvarez-Hamelin
& Busch, *A low complexity visualization tool that helps to perform complex systems
analysis*, New J. Phys. 10 (2008) 125003: a lower bound of the edge connectivity of
the nodes of each shell, built from the clusters of the nested component tree.

A **cluster** is a connected piece of a shell inside a component (the ones the layout
draws, in the same order). The walk grows a *k-connected set* ``C`` from the top
shell down:

1. **Seed.** From the top shell down, the first cluster ``Q`` whose induced subgraph
   has diameter at most 2 (*wide*: or a minimum edge cut of at least the shell index;
   *strict*: and a minimum degree of at least the shell index) joins ``C`` with
   k-connectivity equal to its shell index.
2. **Growth.** For every remaining shell ``k`` from the top down, a cluster ``Q`` joins
   ``C`` with k-connectivity ``k`` when the graph of ``Q`` with ``C`` contracted to a
   single vertex has diameter at most 2 (skipped for ``k = 1``) and either at least
   ``k`` nodes of ``Q`` touch ``C``, or all of them do, or ``phi(Q) >= k`` with ``phi``
   the sum over the nodes of ``Q`` of ``min(max(1, |N(v) ∩ NotB2|), |N(v) ∩ C|)``, where
   ``NotB2`` holds the nodes of ``Q`` with fewer than two neighbors in ``C``.
   Each shell's clusters are examined once, in order, with ``C`` growing as clusters
   are accepted.

The two types differ in what happens to the clusters that are skipped: *wide* (the
C++ default) keeps them pending and tries them again at every lower ``k`` (so a node
can end up with a k-connectivity below its shell index); *strict* drops them.

Nodes that never join ``C`` have k-connectivity 0 and are drawn black on white / white
on black (as squares in the ``bw`` and ``bwi`` schemes), as the C++ did.

Deliberate differences from the C++: shells with no clusters are skipped (the C++
dereferenced an empty list there); the ``shell == 77`` escape in the strict seed is a
debugging leftover and is not reproduced; self-loops are ignored.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping

import networkx as nx

from lanet_vi.decomposition.kcores import _without_self_loops
from lanet_vi.logging_config import get_logger

logger = get_logger(__name__)

#: The vertex that stands for the k-connected set in the contracted cluster graph
_CONTRACTED = object()


def _within_two_hops_of_everything(adjacency: Mapping[Hashable, set[Hashable]]) -> bool:
    """Whether every vertex reaches every other in at most two steps (``computeDiameter2``).

    ``adjacency`` maps each vertex to its *closed* neighborhood (itself included).
    """
    size = len(adjacency)
    for closed in adjacency.values():
        if len(closed) == size:
            continue
        reach: set[Hashable] = set()
        for w in closed:
            reach |= adjacency[w]
        if len(reach) < size:
            return False
    return True


def _closed_neighborhoods(graph: nx.Graph, nodes: Iterable[int]) -> dict[Hashable, set[Hashable]]:
    """Return the closed neighborhoods of the subgraph induced by ``nodes``."""
    members = set(nodes)
    return {v: {w for w in graph[v] if w in members} | {v} for v in members}


def diameter_at_most_two(graph: nx.Graph, nodes: Iterable[int]) -> bool:
    """Whether the subgraph induced by ``nodes`` has diameter at most 2 (connected)."""
    return _within_two_hops_of_everything(_closed_neighborhoods(graph, nodes))


def _min_cut_at_least(graph: nx.Graph, cluster: list[int], k: int) -> bool:
    """Whether the minimum edge cut of the induced subgraph is at least ``k``.

    The C++ read it off a Gomory-Hu tree; a single vertex has no cut (``-1``).
    """
    if len(cluster) < 2:
        return False
    return int(nx.edge_connectivity(graph.subgraph(cluster))) >= k


def _min_degree_at_least(graph: nx.Graph, cluster: list[int], k: int) -> bool:
    """Whether every node of the cluster has at least ``k`` neighbors inside it."""
    members = set(cluster)
    return all(sum(1 for w in graph[v] if w in members) >= k for v in cluster)


def cluster_conditions(
    graph: nx.Graph, cluster: list[int], k: int, kconnectivity: Mapping[int, int]
) -> bool:
    """Test whether ``cluster`` may join the k-connected set at ``k`` (``conditions``).

    Parameters
    ----------
    graph : nx.Graph
        The network
    cluster : List[int]
        Nodes of the cluster
    k : int
        The shell index being processed (``k_conditions``)
    kconnectivity : Mapping[int, int]
        Current k-connectivity of every node; non-zero means "in ``C``"

    Returns
    -------
    bool
        Whether the contracted diameter and the frontier / ``phi`` conditions hold
    """
    # Neighbors in C of every node (evaluateCutC)
    cut_c = {v: sum(1 for w in graph[v] if kconnectivity.get(w, 0)) for v in cluster}

    # First condition: the cluster with C contracted to one vertex has diameter <= 2.
    # The vertex exists only when some node touches C (it is created by its edges).
    if k != 1:
        adjacency = _closed_neighborhoods(graph, cluster)
        touching = [v for v in cluster if cut_c[v] >= 1]
        if touching:
            adjacency[_CONTRACTED] = set(touching) | {_CONTRACTED}
            for v in touching:
                adjacency[v].add(_CONTRACTED)
        if not _within_two_hops_of_everything(adjacency):
            return False

    # Second condition: the frontier B (nodes touching C) is large enough, or phi is
    frontier = sum(1 for v in cluster if cut_c[v] >= 1)
    if frontier >= k or frontier == len(cluster):
        return True
    not_b2 = {v for v in cluster if cut_c[v] < 2}
    phi = 0
    for v in cluster:
        cut_not_b2 = sum(1 for w in graph[v] if w in not_b2)
        phi += min(max(1, cut_not_b2), cut_c[v])
    return phi >= k


def _accept(cluster: list[int], k: int, kconnectivity: dict[int, int]) -> None:
    """Join a cluster to C with k-connectivity ``k`` (``setClusterConnectivity``)."""
    for v in cluster:
        kconnectivity[v] = k


def _growth_pass(
    graph: nx.Graph, clusters: list[list[int]], k: int, kconnectivity: dict[int, int]
) -> list[list[int]]:
    """Examine the clusters once, in order, accepting those that pass; return the rest.

    The C++ scanned a copy of the list, popped it up to the accepted cluster and
    rescanned from there, i.e. one forward pass with C growing on the way.
    """
    left: list[list[int]] = []
    for cluster in clusters:
        if cluster_conditions(graph, cluster, k, kconnectivity):
            _accept(cluster, k, kconnectivity)
        else:
            left.append(cluster)
    return left


def _seed(
    graph: nx.Graph,
    remaining: dict[int, list[list[int]]],
    kconnectivity: dict[int, int],
    *,
    strict: bool,
) -> dict[int, list[list[int]]]:
    """Find the first seed cluster from the top shell down.

    Every shell visited (the seed's included) is removed from ``remaining`` and returned
    with its unseeded clusters, shell by shell, in the order they were visited.
    """
    visited: dict[int, list[list[int]]] = {}
    for shell in sorted(remaining, reverse=True):
        clusters = remaining.pop(shell)
        for i, cluster in enumerate(clusters):
            if strict:
                ok = diameter_at_most_two(graph, cluster) and _min_degree_at_least(
                    graph, cluster, shell
                )
            else:
                ok = diameter_at_most_two(graph, cluster) or _min_cut_at_least(
                    graph, cluster, shell
                )
            if ok:
                _accept(cluster, shell, kconnectivity)
                visited[shell] = clusters[:i] + clusters[i + 1 :]
                return visited
        visited[shell] = clusters
    return visited


def compute_kconnectivity(
    graph: nx.Graph,
    clusters_by_shell: Mapping[int, list[list[int]]],
    kind: str = "wide",
) -> dict[int, int]:
    """
    K-connectivity of every node from the clusters of the k-core component tree.

    Parameters
    ----------
    graph : nx.Graph
        The network (undirected, simple, unweighted: the C++ refused anything else)
    clusters_by_shell : Mapping[int, List[List[int]]]
        The clusters of each shell index in the order of the component tree walk
        (:func:`lanet_vi.visualization.lanet_layout.clusters_by_index`)
    kind : str
        ``"wide"`` (the C++ default) or ``"strict"``

    Returns
    -------
    Dict[int, int]
        K-connectivity of every node of the graph; 0 for the nodes that are not
        k-connected

    Raises
    ------
    ValueError
        If ``kind`` is not ``"wide"`` or ``"strict"``

    Examples
    --------
    >>> root = build_component_tree(G, cores, edge_index, rng)
    >>> kconn = compute_kconnectivity(G, clusters_by_index(root))
    """
    if kind not in ("wide", "strict"):
        raise ValueError(f"kind must be 'wide' or 'strict', not {kind!r}")
    graph = _without_self_loops(graph, "the k-connectivity")
    strict = kind == "strict"
    kconnectivity: dict[int, int] = {}
    remaining = {shell: list(clusters) for shell, clusters in clusters_by_shell.items()}

    # Part A: the seed. Wide keeps the clusters it skipped for later; strict drops them.
    pending = _seed(graph, remaining, kconnectivity, strict=strict)
    if strict:
        pending = {}

    # Part B: every remaining shell from the top down
    while remaining:
        k = max(remaining)
        if not strict:
            # Clusters of higher shells still pending get another chance at this k
            for shell in sorted(pending, reverse=True):
                if shell < k:
                    break
                pending[shell] = _growth_pass(graph, pending[shell], k, kconnectivity)
        left = _growth_pass(graph, remaining.pop(k), k, kconnectivity)
        if not strict:
            pending[k] = left

    result = {v: kconnectivity.get(v, 0) for v in graph.nodes()}
    connected = sum(1 for value in result.values() if value)
    logger.info(f"K-connectivity ({kind}): {connected} of {len(result)} nodes are k-connected")
    return result
