"""Tests for Network class."""

import tempfile
from pathlib import Path

import networkx as nx
import pytest

from lanet_vi.core.network import Network
from lanet_vi.models.config import DecompositionType, LaNetConfig


def test_network_initialization():
    """Test Network initialization."""
    G = nx.karate_club_graph()
    net = Network(G)

    assert net.graph is not None
    assert net.config is not None
    assert net.decomposition is None


def test_network_from_edge_list():
    """Test loading network from edge list."""
    # Create temporary edge list file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("0 1\n")
        f.write("1 2\n")
        f.write("2 0\n")
        temp_path = f.name

    try:
        net = Network.from_edge_list(temp_path)
        assert net.graph.number_of_nodes() == 3
        assert net.graph.number_of_edges() == 3
    finally:
        Path(temp_path).unlink()


def test_network_decompose_kcores():
    """Test k-core decomposition."""
    G = nx.karate_club_graph()
    net = Network(G)

    result = net.decompose(DecompositionType.KCORES)

    assert result is not None
    assert result.decomp_type == "kcores"
    assert net.decomposition is result


def test_network_decompose_kdenses():
    """Test k-dense decomposition."""
    G = nx.karate_club_graph()
    net = Network(G)

    result = net.decompose(DecompositionType.KDENSES)

    assert result is not None
    assert result.decomp_type == "kdenses"
    assert net.decomposition is result


def test_network_compute_layout():
    """Test layout computation."""
    G = nx.karate_club_graph()
    net = Network(G)
    net.decompose()

    layout = net.compute_layout()

    assert layout is not None
    assert len(layout.node_positions) > 0
    assert len(layout.node_colors) > 0
    assert len(layout.node_sizes) > 0


def test_network_compute_layout_without_decomposition():
    """Test that layout fails without decomposition."""
    G = nx.karate_club_graph()
    net = Network(G)

    with pytest.raises(ValueError, match="Must call decompose"):
        net.compute_layout()


def test_network_get_metadata():
    """Test metadata retrieval."""
    G = nx.karate_club_graph()
    net = Network(G)

    metadata = net.get_metadata()

    assert metadata["num_nodes"] == 34
    assert metadata["num_edges"] == 78
    assert metadata["max_degree"] > 0
    assert metadata["avg_degree"] > 0
