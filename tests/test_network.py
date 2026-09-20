"""Tests for Network class."""

import itertools
import tempfile
from pathlib import Path

import networkx as nx
import pytest

from lanet_vi.core.network import Network
from lanet_vi.models.config import (
    DecompositionConfig,
    DecompositionType,
    GraphConfig,
    LaNetConfig,
)


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


def _config(**decomposition) -> LaNetConfig:
    return LaNetConfig(decomposition=DecompositionConfig(**decomposition))


def test_from_layer_keeps_the_induced_subgraph_and_the_kcore_indices(karate: nx.Graph):
    """k-cores: the nodes of index >= K keep their indices; the pipeline runs on them."""
    plain = nx.Graph(karate.edges())
    full = Network(plain).decompose()
    expected_nodes = {node for node, index in full.node_indices.items() if index >= 3}
    assert 0 < len(expected_nodes) < plain.number_of_nodes()

    net = Network(plain, _config(from_layer=3))
    result = net.decompose()

    assert set(net.graph.nodes()) == expected_nodes
    assert set(net.graph.edges()) == set(plain.subgraph(expected_nodes).edges())
    assert result.node_indices == {node: full.node_indices[node] for node in expected_nodes}
    assert result.min_index == 3
    assert {node for component in result.components for node in component.nodes} == expected_nodes
    layout = net.compute_layout()
    assert set(layout.node_positions) == expected_nodes
    assert net.get_metadata()["num_nodes"] == len(expected_nodes)


def test_from_layer_weighted_reuses_the_strength_intervals(karate: nx.Graph):
    """Weighted k-cores re-peel with the whole graph's p-function; indices are kept."""
    config = LaNetConfig(
        graph=GraphConfig(weighted=True),
        decomposition=DecompositionConfig(granularity=5),
    )
    full = Network(karate, config).decompose()
    assert full.p_function is not None
    layer = full.max_index - 1
    expected_nodes = {node for node, index in full.node_indices.items() if index >= layer}

    config.decomposition.from_layer = layer
    net = Network(karate, config)
    result = net.decompose()

    assert set(net.graph.nodes()) == expected_nodes
    assert result.p_function == full.p_function
    assert result.node_indices == {node: full.node_indices[node] for node in expected_nodes}


def test_from_layer_kdenses_recomputes_on_the_induced_subgraph(karate: nx.Graph):
    """k-denses: indices are recomputed on the induced subgraph (never above the old)."""
    plain = nx.Graph(karate.edges())
    full = Network(plain, _config(decomp_type=DecompositionType.KDENSES)).decompose()
    layer = 4
    expected_nodes = {node for node, index in full.node_indices.items() if index >= layer}
    assert 0 < len(expected_nodes) < plain.number_of_nodes()

    net = Network(plain, _config(decomp_type=DecompositionType.KDENSES, from_layer=layer))
    result = net.decompose()

    assert set(net.graph.nodes()) == expected_nodes
    assert set(result.node_indices) == expected_nodes
    assert all(result.node_indices[node] <= full.node_indices[node] for node in expected_nodes)
    assert set(result.metadata["edge_indices"]) == {
        (min(u, v), max(u, v)) for u, v in net.graph.edges()
    }


def test_from_layer_dcores_uses_the_ring_index():
    """d-cores: K applies to max(k_in, k_out), the index the rings are drawn by."""
    digraph = nx.DiGraph()
    digraph.add_edges_from(itertools.permutations(range(4), 2))  # a 4-clique both ways
    digraph.add_edges_from([(4, 0), (5, 1), (4, 5)])  # a fringe of index 1
    full = Network(digraph, _config(decomp_type=DecompositionType.DCORES)).decompose()
    assert full.max_index == 3

    net = Network(digraph, _config(decomp_type=DecompositionType.DCORES, from_layer=2))
    result = net.decompose()

    assert set(net.graph.nodes()) == {0, 1, 2, 3}
    assert set(result.node_indices.values()) == {3}


def test_from_layer_above_the_maximum_index_is_an_error(karate: nx.Graph):
    """A layer nobody reaches leaves an empty graph: refused with a clear message."""
    plain = nx.Graph(karate.edges())
    net = Network(plain, _config(from_layer=99))
    with pytest.raises(ValueError, match="from_layer=99 leaves no node.*maximum index is 4"):
        net.decompose()
    assert net.graph.number_of_nodes() == plain.number_of_nodes()  # untouched
