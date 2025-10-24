"""Tests for k-dense decomposition."""

import networkx as nx

from lanet_vi.decomposition.kdenses import compute_kdenses


def test_kdenses_simple_graph():
    """Test k-dense decomposition on a simple graph."""
    # Create a graph with triangles
    G = nx.Graph()
    # Triangle
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    # Another triangle sharing an edge
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])

    result = compute_kdenses(G)

    assert result.decomp_type == "kdenses"
    assert result.max_index >= 2  # Minimum dense index
    assert len(result.node_indices) == 4


def test_kdenses_no_triangles():
    """Test k-dense on graph without triangles."""
    # Create a tree (no triangles)
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (1, 3)])

    result = compute_kdenses(G)

    assert result.decomp_type == "kdenses"
    # All nodes should have minimum dense index (2) since no triangles
    assert all(idx == 2 for idx in result.node_indices.values())


def test_kdenses_karate_club():
    """Test k-dense on Zachary's karate club graph."""
    G = nx.karate_club_graph()

    result = compute_kdenses(G)

    assert result.decomp_type == "kdenses"
    assert result.max_index >= 2
    assert len(result.node_indices) == 34


def test_kdenses_empty_graph():
    """Test k-dense on empty graph."""
    G = nx.Graph()

    result = compute_kdenses(G)

    assert result.decomp_type == "kdenses"
    assert len(result.node_indices) == 0
