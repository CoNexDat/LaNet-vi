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
