"""Tests for k-core decomposition."""

import networkx as nx

from lanet_vi.decomposition.kcores import compute_kcores
from lanet_vi.models.config import DecompositionConfig


def test_kcores_simple_graph():
    """Test k-core decomposition on a simple graph."""
    # Create a simple graph with known k-core structure
    G = nx.Graph()
    # Triangle (3-core)
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    # Attached nodes (lower cores)
    G.add_edge(0, 3)
    G.add_edge(3, 4)

    result = compute_kcores(G)

    assert result.decomp_type == "kcores"
    assert result.max_index >= 2  # Triangle has at least 2-core
    assert result.min_index >= 0
    assert len(result.node_indices) == 5  # 5 nodes total


def test_kcores_karate_club():
    """Test k-core on Zachary's karate club graph."""
    G = nx.karate_club_graph()

    result = compute_kcores(G)

    assert result.decomp_type == "kcores"
    assert result.max_index > 0
    assert len(result.node_indices) == 34  # Karate club has 34 nodes


def test_kcores_empty_graph():
    """Test k-core on empty graph."""
    G = nx.Graph()

    result = compute_kcores(G)

    assert result.decomp_type == "kcores"
    assert result.max_index == 0
    assert result.min_index == 0
    assert len(result.node_indices) == 0


def test_kcores_weighted_graph():
    """Test k-core on weighted graph."""
    G = nx.Graph()
    G.add_edge(0, 1, weight=2.0)
    G.add_edge(1, 2, weight=3.0)
    G.add_edge(2, 0, weight=1.5)

    config = DecompositionConfig(granularity=3)
    result = compute_kcores(G, config)

    assert result.decomp_type == "kcores"
    assert result.p_function is not None  # Weighted graph should have p-function
    assert len(result.p_function) > 0
