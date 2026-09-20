"""Tests for d-core decomposition on directed graphs."""

import networkx as nx
import pytest

from lanet_vi.decomposition.dcores import compute_dcore_table, compute_dcores


def test_dcores_directed_triangle():
    """A directed 3-cycle gives every node a (1, 1) core pair."""
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])

    result = compute_dcores(G)

    assert result.decomp_type == "dcores"
    assert len(result.node_indices) == 3
    assert result.max_index >= 1


def test_dcores_rejects_undirected_graph():
    """Undirected input is an error."""
    with pytest.raises(ValueError):
        compute_dcores(nx.karate_club_graph())  # type: ignore[arg-type]


def test_dcores_empty_graph():
    """An empty directed graph decomposes to nothing."""
    result = compute_dcores(nx.DiGraph())

    assert result.decomp_type == "dcores"
    assert len(result.node_indices) == 0
    assert result.max_index == 0
    assert result.min_index == 0


def test_dcores_min_index_reflects_isolated_nodes():
    """An isolated node has core 0, so min_index is 0 rather than a hard-coded 1."""
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    G.add_node(99)

    result = compute_dcores(G)

    assert result.node_indices[99] == 0
    assert result.min_index == 0
    assert result.max_index >= 1


def test_dcores_star_has_lower_core_than_cycle():
    """Nodes in a directed cycle sit deeper than leaves of an attached star."""
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])  # cycle
    G.add_edges_from([(0, 10), (0, 11), (0, 12)])  # out-star from node 0

    result = compute_dcores(G)

    assert result.node_indices[10] <= result.node_indices[0]


def test_find_components_by_dcore_groups_by_level():
    """Components are built with the model's field names and grouped by max(k_in, k_out)."""
    from lanet_vi.decomposition.dcores import find_components_by_dcore

    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])  # cycle: level 1
    G.add_edge(5, 6)  # detached chain: both peel away, level 0
    G.add_node(99)  # isolated: level 0

    result = find_components_by_dcore(G, compute_dcores(G))

    assert {c.shell_index for c in result.components} == {0, 1}
    top = [c for c in result.components if c.shell_index == 1]
    assert len(top) == 1 and sorted(top[0].nodes) == [0, 1, 2]
    bottom = sorted(sorted(c.nodes) for c in result.components if c.shell_index == 0)
    assert bottom == [[5, 6], [99]]
    assert sum(c.size for c in result.components) == G.number_of_nodes()
    assert [c.component_id for c in result.components] == list(range(len(result.components)))


def test_network_decompose_dcores_end_to_end():
    """Network.decompose(DCORES) no longer raises when building components."""
    from lanet_vi.core.network import Network
    from lanet_vi.models.config import DecompositionType, GraphConfig, LaNetConfig

    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 3), (1, 4)])
    net = Network(G, config=LaNetConfig(graph=GraphConfig(directed=True)))

    result = net.decompose(DecompositionType.DCORES)

    assert result.decomp_type == "dcores"
    assert result.components
    assert set(result.metadata["d_cores"]) == set(G.nodes())
    assert sum(c.size for c in result.components) == G.number_of_nodes()


def _kl_core(graph: nx.DiGraph, k: int, out_min: int) -> set[int]:
    """Compute the (k, l)-core by repeated removal, straight from the definition."""
    core = graph.copy()
    while True:
        drop = [v for v in core if core.in_degree(v) < k or core.out_degree(v) < out_min]
        if not drop:
            return set(core)
        core.remove_nodes_from(drop)


def _brute_force_table(graph: nx.DiGraph) -> dict[int, dict[int, int]]:
    """Compute {l: {node: max k with node in the (k, l)-core}} by testing every (k, l)."""
    table = {}
    for out_min in range(max((d for _, d in graph.out_degree()), default=0) + 2):
        members = _kl_core(graph, 0, out_min)
        if not members:
            break
        row = {}
        for node in members:
            k = 0
            while node in _kl_core(graph, k + 1, out_min):
                k += 1
            row[node] = k
        table[out_min] = row
    return table


def test_dcore_table_matches_the_definition_on_random_digraphs():
    """The peeling agrees with the brute-force (k, l)-core table, self-loops included."""
    import random

    for seed in range(40):
        rng = random.Random(seed)
        n, p = rng.randint(2, 12), rng.uniform(0.05, 0.6)
        graph = nx.gnp_random_graph(n, p, seed=seed, directed=True)
        if seed % 5 == 0:
            graph.add_edge(0, 0)  # a self-loop counts for neither degree
        graph_no_loops = nx.DiGraph(graph)
        graph_no_loops.remove_edges_from(nx.selfloop_edges(graph_no_loops))
        assert compute_dcore_table(graph) == _brute_force_table(graph_no_loops), seed


def test_dcore_table_row_zero_is_the_in_core_and_rows_are_nested():
    """Row 0 equals k_in of compute_dcores; the (0, l)-cores shrink with l."""
    graph = nx.gnp_random_graph(30, 0.15, seed=3, directed=True)
    table = compute_dcore_table(graph)
    pairs = compute_dcores(graph).metadata["d_cores"]
    assert table[0] == {node: k_in for node, (k_in, _) in pairs.items()}
    max_out = max(k_out for _, k_out in pairs.values())
    assert sorted(table) == list(range(max_out + 1))
    for out_min in range(1, max_out + 1):
        assert set(table[out_min]) <= set(table[out_min - 1])
        # k can only drop when l grows (more constraints)
        assert all(table[out_min][v] <= table[out_min - 1][v] for v in table[out_min])


def test_dcore_table_edge_cases():
    """Undirected graphs are refused; empty and edgeless graphs give the trivial tables."""
    with pytest.raises(ValueError, match="directed"):
        compute_dcore_table(nx.Graph([(0, 1)]))
    assert compute_dcore_table(nx.DiGraph()) == {}
    edgeless = nx.DiGraph()
    edgeless.add_nodes_from([1, 2])
    assert compute_dcore_table(edgeless) == {0: {1: 0, 2: 0}}
    # A directed cycle: every node has in-degree 1 and out-degree 1
    assert compute_dcore_table(nx.DiGraph([(0, 1), (1, 2), (2, 0)])) == {
        0: {0: 1, 1: 1, 2: 1},
        1: {0: 1, 1: 1, 2: 1},
    }
