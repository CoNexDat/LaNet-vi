"""Tests for d-core decomposition on directed graphs."""

import networkx as nx
import pytest

from lanet_vi.decomposition.dcores import compute_dcores


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
