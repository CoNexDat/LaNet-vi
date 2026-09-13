"""Tests for k-dense decomposition."""

import json

import networkx as nx
import pytest

from lanet_vi.decomposition.kdenses import compute_kdenses, find_components_by_dense
from lanet_vi.io.writers import write_decomposition_json


def _brute_force_kdense(graph: nx.Graph) -> dict[int, int]:
    """K-dense indices by repeated k-truss peeling, independent of the production code.

    An edge has index ``k`` when it survives in the k-truss (every edge closes at least
    ``k - 2`` triangles) but not in the ``(k + 1)``-truss. A vertex takes the maximum
    over its incident edges, or 2 if none of them lie in a triangle.
    """
    remaining = nx.Graph(graph)
    remaining.remove_edges_from(nx.selfloop_edges(remaining))
    node_index = {node: 2 for node in graph.nodes()}
    k = 2
    while remaining.number_of_edges():
        # Strip everything that is not in the (k + 1)-truss; what falls out has index k.
        changed = True
        while changed:
            changed = False
            for u, v in list(remaining.edges()):
                if len(set(remaining[u]) & set(remaining[v])) < k - 1:
                    remaining.remove_edge(u, v)
                    node_index[u] = max(node_index[u], k)
                    node_index[v] = max(node_index[v], k)
                    changed = True
        k += 1
    return node_index


def test_kdenses_simple_graph():
    """Two triangles sharing an edge: every edge closes exactly one triangle."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])

    result = compute_kdenses(G)

    assert result.decomp_type == "kdenses"
    assert result.node_indices == {0: 3, 1: 3, 2: 3, 3: 3}
    assert result.max_index == 3
    assert result.min_index == 3
    assert result.metadata["edge_indices"] == {
        (0, 1): 3,
        (0, 2): 3,
        (1, 2): 3,
        (1, 3): 3,
        (2, 3): 3,
    }


def test_kdenses_pendant_edge_has_index_two():
    """An edge outside every triangle keeps index 2; its endpoint too if it has no other."""
    G = nx.Graph([(0, 1), (1, 2), (0, 2), (2, 3)])

    result = compute_kdenses(G)

    assert result.node_indices == {0: 3, 1: 3, 2: 3, 3: 2}
    assert result.metadata["edge_indices"][(2, 3)] == 2


def test_kdenses_independent_of_edge_order():
    """Issue #19: the triangle (0,2),(0,1),(1,2) used to come out as all 2s."""
    first = nx.Graph([(0, 2), (0, 1), (1, 2)])
    second = nx.Graph([(0, 1), (0, 2), (1, 2)])

    assert compute_kdenses(first).node_indices == {0: 3, 1: 3, 2: 3}
    assert compute_kdenses(second).node_indices == {0: 3, 1: 3, 2: 3}


def test_kdenses_complete_graph():
    """In K_n every edge closes n - 2 triangles, so every vertex has index n."""
    for n in (3, 4, 6):
        result = compute_kdenses(nx.complete_graph(n))
        assert set(result.node_indices.values()) == {n}


def test_kdenses_is_not_the_dual_vertex_core():
    """Pair peeling (C++) differs from a plain k-core of the edge/triangle dual graph."""
    # Node 2 touches four triangles, all through node 1: (1,2,3), (1,2,5), (1,2,6), (1,2,7).
    # Its side edges (2,3), (2,5), (2,6), (2,7) close one triangle each and peel first, which
    # leaves (1,2) with no triangle at all, so node 2 gets index 3. A vertex k-core of the
    # dual graph keeps (1,2) at 4 through the (1,x) edges that survive in the dense rest.
    G = nx.Graph(
        [(0, 1), (0, 3), (0, 7), (0, 8), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7)]
        + [(1, 8), (2, 3), (2, 5), (2, 6), (2, 7), (3, 8), (4, 5), (4, 6), (4, 8), (5, 8)]
        + [(6, 8), (7, 8)]
    )

    result = compute_kdenses(G)

    assert result.node_indices == _brute_force_kdense(G)
    assert result.node_indices[2] == 3
    assert result.metadata["edge_indices"][(1, 2)] == 3
    assert all(idx == 4 for node, idx in result.node_indices.items() if node != 2)


@pytest.mark.parametrize("seed", range(40))
def test_kdenses_matches_brute_force_truss(seed: int):
    """Random graphs agree with an independent k-truss peeling (issue #19)."""
    G = nx.gnp_random_graph(30, 0.25, seed=seed)

    result = compute_kdenses(G)

    assert result.node_indices == _brute_force_kdense(G)


def test_kdenses_multigraph_and_self_loops():
    """Parallel edges and self-loops do not create spurious triangles."""
    multi = nx.MultiGraph([(0, 1), (0, 1), (1, 2), (0, 2), (2, 2), (2, 3), (3, 3)])

    result = compute_kdenses(multi)

    assert result.node_indices == {0: 3, 1: 3, 2: 3, 3: 2}
    assert set(result.metadata["edge_indices"]) == {(0, 1), (0, 2), (1, 2), (2, 3)}


def test_kdenses_no_triangles():
    """Test k-dense on graph without triangles."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (1, 3)])

    result = compute_kdenses(G)

    assert result.decomp_type == "kdenses"
    assert all(idx == 2 for idx in result.node_indices.values())
    assert all(idx == 2 for idx in result.metadata["edge_indices"].values())


def test_kdenses_karate_club():
    """Test k-dense on Zachary's karate club graph."""
    G = nx.karate_club_graph()

    result = compute_kdenses(G)

    assert result.decomp_type == "kdenses"
    assert result.max_index == 5
    assert result.min_index == 2
    assert len(result.node_indices) == 34
    assert result.node_indices == _brute_force_kdense(G)


def test_kdenses_empty_graph():
    """Test k-dense on empty graph."""
    G = nx.Graph()

    result = compute_kdenses(G)

    assert result.decomp_type == "kdenses"
    assert len(result.node_indices) == 0
    assert result.max_index == 2


def test_kdenses_rejects_directed():
    """Directed graphs are refused."""
    with pytest.raises(ValueError, match="undirected"):
        compute_kdenses(nx.DiGraph([(0, 1)]))


def test_kdenses_components_and_json_export(tmp_path):
    """Components are grouped per dense index and edge indices survive the JSON writer."""
    G = nx.Graph([(0, 1), (1, 2), (0, 2), (2, 3)])
    result = find_components_by_dense(G, compute_kdenses(G))

    assert [(c.dense_index, sorted(c.nodes)) for c in result.components] == [
        (3, [0, 1, 2]),
        (2, [3]),
    ]

    out = tmp_path / "kdenses.json"
    write_decomposition_json(result, out)
    data = json.loads(out.read_text())
    assert sorted(data["metadata"]["edge_indices"]) == [
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
        [2, 3, 2],
    ]
