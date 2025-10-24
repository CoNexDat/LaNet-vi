"""K-dense decomposition using triangle-based dual graph approach."""

from typing import Dict, List, Tuple

import networkx as nx

from lanet_vi.models.graph import Component, DecompositionResult


def compute_kdenses(graph: nx.Graph) -> DecompositionResult:
    """
    Compute k-dense decomposition of a graph.

    K-dense decomposition is based on triangle density. It constructs a dual graph
    where edges become vertices and triangles become edges, then applies k-core
    decomposition to the dual graph.

    Parameters
    ----------
    graph : nx.Graph
        Input graph (must be undirected)

    Returns
    -------
    DecompositionResult
        K-dense decomposition results with dense indices

    Notes
    -----
    The k-dense index of a vertex is computed as:
    dense_index = max(k-core of adjacent edges in dual graph) / 2 + 2

    Examples
    --------
    >>> G = nx.karate_club_graph()
    >>> result = compute_kdenses(G)
    >>> print(f"Max dense index: {result.max_index}")
    """
    if graph.is_directed():
        raise ValueError("K-dense decomposition requires an undirected graph")

    # Build dual graph of triangles
    dual_graph, edge_to_vertex_map = _build_triangle_dual_graph(graph)

    # Apply k-core decomposition to dual graph
    if dual_graph.number_of_nodes() > 0:
        dual_cores = nx.core_number(dual_graph)
    else:
        dual_cores = {}

    # Map k-dense values back to original vertices
    dense_indices = _compute_vertex_dense_indices(graph, edge_to_vertex_map, dual_cores)

    return DecompositionResult(
        decomp_type="kdenses",
        node_indices=dense_indices,
        max_index=max(dense_indices.values()) if dense_indices else 2,
        min_index=min(dense_indices.values()) if dense_indices else 2,
    )


def _build_triangle_dual_graph(
    graph: nx.Graph,
) -> Tuple[nx.Graph, Dict[Tuple[int, int], int]]:
    """
    Build dual graph where edges are vertices and triangles are edges.

    Parameters
    ----------
    graph : nx.Graph
        Original graph

    Returns
    -------
    dual_graph : nx.Graph
        Dual graph
    edge_to_vertex_map : Dict[Tuple[int, int], int]
        Mapping from original edge (u, v) to dual vertex ID
    """
    dual_graph = nx.Graph()
    edge_to_vertex_map: Dict[Tuple[int, int], int] = {}
    edge_counter = 0

    # Find all triangles using NetworkX (for diagnostics)
    _triangles = nx.triangles(graph)  # Reserved for future validation

    # Process each node's triangles
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))

        # Check each pair of neighbors for triangles
        for i, neighbor1 in enumerate(neighbors):
            if neighbor1 <= node:
                continue

            for neighbor2 in neighbors[i + 1 :]:
                if neighbor2 <= neighbor1:
                    continue

                # Check if this forms a triangle
                if graph.has_edge(neighbor1, neighbor2):
                    # Triangle found: node - neighbor1 - neighbor2
                    v1, v2, v3 = sorted([node, neighbor1, neighbor2])

                    # Get or create dual vertices for each edge
                    edge1 = _get_or_create_dual_vertex(
                        edge_to_vertex_map, (v1, v2), edge_counter
                    )
                    if edge1 == edge_counter:
                        edge_counter += 1

                    edge2 = _get_or_create_dual_vertex(
                        edge_to_vertex_map, (v2, v3), edge_counter
                    )
                    if edge2 == edge_counter:
                        edge_counter += 1

                    edge3 = _get_or_create_dual_vertex(
                        edge_to_vertex_map, (v1, v3), edge_counter
                    )
                    if edge3 == edge_counter:
                        edge_counter += 1

                    # Add edges in dual graph (triangle becomes edges between its 3 edges)
                    dual_graph.add_edge(edge1, edge2)
                    dual_graph.add_edge(edge2, edge3)
                    dual_graph.add_edge(edge1, edge3)

    return dual_graph, edge_to_vertex_map


def _get_or_create_dual_vertex(
    edge_map: Dict[Tuple[int, int], int],
    edge: Tuple[int, int],
    counter: int,
) -> int:
    """
    Get existing dual vertex ID or create new one for an edge.

    Parameters
    ----------
    edge_map : Dict[Tuple[int, int], int]
        Current edge to vertex mapping
    edge : Tuple[int, int]
        Edge (u, v) with u < v
    counter : int
        Next available vertex ID

    Returns
    -------
    int
        Dual vertex ID for this edge
    """
    # Ensure edge is in canonical form (u < v)
    u, v = edge
    if u > v:
        u, v = v, u
    edge = (u, v)

    if edge not in edge_map:
        edge_map[edge] = counter
        return counter
    return edge_map[edge]


def _compute_vertex_dense_indices(
    graph: nx.Graph,
    edge_to_vertex_map: Dict[Tuple[int, int], int],
    dual_cores: Dict[int, int],
) -> Dict[int, int]:
    """
    Compute dense index for each vertex in original graph.

    The dense index of a vertex is the maximum k-core value of its
    incident edges in the dual graph, divided by 2 and offset by 2.

    Parameters
    ----------
    graph : nx.Graph
        Original graph
    edge_to_vertex_map : Dict[Tuple[int, int], int]
        Mapping from edge to dual vertex
    dual_cores : Dict[int, int]
        K-core numbers in dual graph

    Returns
    -------
    Dict[int, int]
        Dense index for each vertex
    """
    dense_indices: Dict[int, int] = {}

    for node in graph.nodes():
        max_edge_core = 0

        # Check all incident edges
        for neighbor in graph.neighbors(node):
            # Get edge in canonical form
            u, v = sorted([node, neighbor])
            edge = (u, v)

            if edge in edge_to_vertex_map:
                dual_vertex = edge_to_vertex_map[edge]
                if dual_vertex in dual_cores:
                    edge_core = dual_cores[dual_vertex]
                    max_edge_core = max(max_edge_core, edge_core)

        # Dense index formula from C++ code
        dense_indices[node] = max_edge_core // 2 + 2

    return dense_indices


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
    denses: Dict[int, List[int]] = {}
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
