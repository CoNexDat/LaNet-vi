"""K-dense (m-core) decomposition by triangle-pair peeling.

This ports ``graph_kdenses.cpp`` / ``graph_triangled_kcores.cpp`` from the C++ LaNet-vi.
The C++ builds a dual graph whose vertices are the edges of the input graph and whose
edges join the three sides of every triangle, then peels it: whenever a dual vertex is
removed, the two other sides of each triangle it belonged to lose one triangle. That is
the k-truss decomposition of the input graph. The k-dense index of an edge is its
trussness (``2`` for edges in no triangle, ``3`` for edges in a triangle whose sides are
in no other triangle, ...) and the k-dense index of a vertex is the maximum over its
incident edges.
"""

import networkx as nx

from lanet_vi.models.graph import Component, DecompositionResult

#: k-dense index of an isolated vertex or an edge that lies in no triangle.
MIN_DENSE_INDEX = 2


def compute_kdenses(graph: nx.Graph) -> DecompositionResult:
    """
    Compute the k-dense decomposition of a graph.

    Parallel edges and self-loops are ignored, as in the C++ reader.

    Parameters
    ----------
    graph : nx.Graph
        Input graph (must be undirected)

    Returns
    -------
    DecompositionResult
        ``node_indices`` maps every node to its k-dense index (``>= 2``);
        ``metadata["edge_indices"]`` maps every simple edge ``(u, v)`` to its k-dense
        index, which the C++ uses to colour edges.

    Notes
    -----
    The k-dense index of an edge is ``2 + s`` where ``s`` is its truss support number:
    the largest ``s`` such that the edge belongs to a subgraph in which every edge closes
    at least ``s`` triangles. The k-dense index of a vertex is the maximum k-dense index
    of its incident edges, or ``2`` if it has none.

    Examples
    --------
    >>> G = nx.karate_club_graph()
    >>> result = compute_kdenses(G)
    >>> print(f"Max dense index: {result.max_index}")
    Max dense index: 5
    """
    if graph.is_directed():
        raise ValueError("K-dense decomposition requires an undirected graph")

    nodes = list(graph.nodes())
    index_of = {node: i for i, node in enumerate(nodes)}
    adjacency = _simple_adjacency(graph, index_of)

    support = _edge_truss_support(adjacency)

    edge_indices: dict[tuple[int, int], int] = {}
    dense_indices: dict[int, int] = {node: MIN_DENSE_INDEX for node in nodes}
    for (u, v), s in support.items():
        dense = MIN_DENSE_INDEX + s
        edge_indices[(nodes[u], nodes[v])] = dense
        if dense > dense_indices[nodes[u]]:
            dense_indices[nodes[u]] = dense
        if dense > dense_indices[nodes[v]]:
            dense_indices[nodes[v]] = dense

    return DecompositionResult(
        decomp_type="kdenses",
        node_indices=dense_indices,
        max_index=max(dense_indices.values()) if dense_indices else MIN_DENSE_INDEX,
        min_index=min(dense_indices.values()) if dense_indices else MIN_DENSE_INDEX,
        metadata={"edge_indices": edge_indices},
    )


def _simple_adjacency(graph: nx.Graph, index_of: dict[int, int]) -> list[set[int]]:
    """
    Build an integer-indexed adjacency of the simple graph underlying ``graph``.

    Parallel edges collapse and self-loops are dropped, so the result matches what the
    C++ reader produced (it always loaded a simple graph before looking for triangles).

    Parameters
    ----------
    graph : nx.Graph
        Input graph (``nx.MultiGraph`` accepted)
    index_of : Dict[int, int]
        Node → position in ``list(graph.nodes())``

    Returns
    -------
    List[Set[int]]
        ``adjacency[i]`` is the set of neighbour positions of node ``i``
    """
    adjacency: list[set[int]] = [set() for _ in index_of]
    for u, v in graph.edges():
        if u == v:
            continue
        iu, iv = index_of[u], index_of[v]
        adjacency[iu].add(iv)
        adjacency[iv].add(iu)
    return adjacency


def _edge_truss_support(adjacency: list[set[int]]) -> dict[tuple[int, int], int]:
    """
    Peel the graph by triangle support (k-truss decomposition).

    This is the algorithm of ``Graph_Triangled_KCores::findTriangledCores``: every edge
    starts with its triangle count, edges are removed in increasing order of support, and
    removing an edge decrements the support of the two other sides of each triangle it
    still closes (never below the support currently being peeled).

    Parameters
    ----------
    adjacency : List[Set[int]]
        Simple, undirected adjacency over integer node ids

    Returns
    -------
    Dict[Tuple[int, int], int]
        Truss support number of every edge ``(u, v)`` with ``u < v``
    """
    support: dict[tuple[int, int], int] = {}
    for u, neighbours in enumerate(adjacency):
        for v in neighbours:
            if u < v:
                support[(u, v)] = len(neighbours & adjacency[v])
    if not support:
        return {}

    # Bucket queue keyed by current support, as in Batagelj-Zaversnik.
    buckets: list[set[tuple[int, int]]] = [set() for _ in range(max(support.values()) + 1)]
    for edge, s in support.items():
        buckets[s].add(edge)

    live = [set(neighbours) for neighbours in adjacency]
    result: dict[tuple[int, int], int] = {}
    for k, bucket in enumerate(buckets):
        while bucket:
            u, v = bucket.pop()
            result[(u, v)] = k
            live[u].discard(v)
            live[v].discard(u)
            # Every remaining common neighbour closed a triangle with (u, v) that is now gone.
            for w in live[u] & live[v]:
                for side in ((u, w) if u < w else (w, u), (v, w) if v < w else (w, v)):
                    s = support[side]
                    if s > k:
                        buckets[s].discard(side)
                        support[side] = s - 1
                        buckets[s - 1].add(side)
    return result


def find_components_by_dense(
    graph: nx.Graph,
    decomposition: DecompositionResult,
) -> DecompositionResult:
    """
    Find connected components for each dense index.

    Parameters
    ----------
    graph : nx.Graph
        Original graph
    decomposition : DecompositionResult
        K-dense decomposition result

    Returns
    -------
    DecompositionResult
        Updated result with component information
    """
    components = []
    component_id = 0

    # Group nodes by dense index
    denses: dict[int, list[int]] = {}
    for node, dense_idx in decomposition.node_indices.items():
        if dense_idx not in denses:
            denses[dense_idx] = []
        denses[dense_idx].append(node)

    # Find components within each dense level
    for dense_idx in sorted(denses.keys(), reverse=True):
        dense_nodes = denses[dense_idx]
        subgraph = graph.subgraph(dense_nodes)

        # Find connected components
        for comp_nodes in nx.connected_components(subgraph):
            components.append(
                Component(
                    component_id=component_id,
                    nodes=list(comp_nodes),
                    dense_index=dense_idx,
                    size=len(comp_nodes),
                )
            )
            component_id += 1

    decomposition.components = components
    return decomposition
