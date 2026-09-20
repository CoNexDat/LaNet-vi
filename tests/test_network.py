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


def test_decompose_twice_with_from_layer_starts_from_the_input_graph(karate: nx.Graph):
    """A second decompose() does not shrink the already extracted layer again."""
    plain = nx.Graph(karate.edges())
    net = Network(plain, _config(from_layer=3))
    first = net.decompose(DecompositionType.KCORES)
    assert net.input_graph is plain

    again = net.decompose(DecompositionType.KCORES)
    assert again.node_indices == first.node_indices

    dense = net.decompose(DecompositionType.KDENSES)
    fresh = Network(plain, _config(decomp_type=DecompositionType.KDENSES, from_layer=3))
    assert dense.node_indices == fresh.decompose().node_indices
    assert set(net.graph.nodes()) == set(fresh.graph.nodes())


def test_detect_communities_runs_on_the_drawn_graph_and_resets_on_decompose(karate: nx.Graph):
    """Communities are those of the layer subgraph, seeded, and cleared by a new decompose()."""
    from lanet_vi.community import detect_communities_louvain
    from lanet_vi.models.config import CommunityConfig

    plain = nx.Graph(karate.edges())
    config = _config(from_layer=3)
    config.community = CommunityConfig(detect_communities=True)
    net = Network(plain, config)
    net.decompose()
    assert net.communities is not None
    assert set(net.communities.node_to_community) == set(net.graph.nodes()) != set(plain)
    expected = detect_communities_louvain(net.graph, seed=config.layout.seed)
    assert net.communities.node_to_community == expected.node_to_community

    # Explicit detection works without the flag; a new decompose() drops the result
    config.community.detect_communities = False
    net.decompose()
    assert net.communities is None
    again = net.detect_communities()
    assert again is net.communities
    assert again.node_to_community == expected.node_to_community
    net.decompose()
    assert net.communities is None


def test_detect_communities_on_directed_and_weighted_graphs():
    """d-core (directed) and weighted graphs are accepted by the detection."""
    from lanet_vi.models.config import CommunityConfig

    digraph = nx.DiGraph([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)])
    config = _config(decomp_type=DecompositionType.DCORES)
    config.graph.directed = True
    config.community = CommunityConfig(detect_communities=True)
    net = Network(digraph, config)
    net.decompose()
    assert net.communities is not None
    assert net.communities.num_communities == 2
    layout = net.compute_layout()
    assert layout.node_colors[0] == layout.node_colors[1] != layout.node_colors[4]

    from networkx.algorithms import community as nx_community

    weighted = nx.Graph()
    weighted.add_weighted_edges_from(
        [(0, 1, 5.0), (1, 2, 5.0), (2, 0, 5.0), (2, 3, 0.1), (3, 4, 5.0), (4, 5, 5.0), (5, 3, 5.0)]
    )
    config = _config()
    config.graph.weighted = True
    config.community = CommunityConfig(detect_communities=True, algorithm="greedy_modularity")
    net = Network(weighted, config)
    net.decompose()
    assert net.communities is not None
    assert net.communities.num_communities == 2
    assert net.communities.get_node_community(3) != net.communities.get_node_community(0)
    # The weights are used: the modularity is the weighted one
    parts = [set(c.nodes) for c in net.communities.communities]
    assert net.communities.modularity == pytest.approx(
        nx_community.modularity(weighted, parts, weight="weight")
    )


def test_kconnectivity_paints_the_nodes_that_are_not_k_connected(karate: nx.Graph):
    """With kconn set, decompose() computes it and the layout marks the nodes left out."""
    from lanet_vi.models.config import BackgroundColor, ColorScheme, KConnectivityType

    plain = nx.Graph(karate.edges())
    config = _config(kconn=True, kconn_type=KConnectivityType.STRICT)
    net = Network(plain, config)
    assert net.kconnectivity is None
    net.decompose()
    assert net.kconnectivity is not None
    assert set(net.kconnectivity) == set(plain)
    # Strict finds no seed on the karate club (its 4-core has diameter 3)
    assert set(net.kconnectivity.values()) == {0}

    # Color: black on the white background, whatever colored the others; no squares in col
    config.visualization.background = BackgroundColor.WHITE
    layout = net.compute_layout()
    assert all(layout.node_colors[v] == (0.0, 0.0, 0.0) for v in plain)
    assert layout.square_nodes == set()
    # Grayscale: squares
    config.visualization.color_scheme = ColorScheme.GRAYSCALE
    config.visualization.background = BackgroundColor.BLACK
    layout = net.compute_layout()
    assert layout.square_nodes == set(plain)
    assert all(layout.node_colors[v] == (1.0, 1.0, 1.0) for v in plain)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "kconn.png"
        net.visualize(out, layout)
        assert out.exists()

    # Wide: everybody is k-connected, the top core at 4; colors are the shell colors again
    config.decomposition.kconn_type = KConnectivityType.WIDE
    config.visualization.color_scheme = ColorScheme.COLOR
    net.decompose()
    assert all(net.kconnectivity[v] > 0 for v in plain)
    top = [v for v, k in net.decomposition.node_indices.items() if k == 4]
    assert all(net.kconnectivity[v] == 4 for v in top)
    layout = net.compute_layout()
    assert layout.square_nodes == set()
    assert len({layout.node_colors[v] for v in plain}) > 1

    # Off again: a new decompose() drops it; an explicit call computes it anyway
    config.decomposition.kconn = False
    net.decompose()
    assert net.kconnectivity is None
    explicit = net.compute_kconnectivity()
    assert explicit is net.kconnectivity
    assert all(explicit[v] == 4 for v in top)


def test_kconnectivity_refuses_what_the_cpp_refused(karate: nx.Graph):
    """No decomposition, k-denses, directed, multigraph and weighted graphs are errors."""
    from lanet_vi.models.config import KConnectivityType

    plain = nx.Graph(karate.edges())
    net = Network(plain)
    with pytest.raises(ValueError, match="decompose"):
        net.compute_kconnectivity()

    net = Network(plain, _config(decomp_type=DecompositionType.KDENSES))
    net.decompose()
    with pytest.raises(ValueError, match="k-core"):
        net.compute_kconnectivity()

    net = Network(karate)  # the weights are autodetected: weighted k-cores
    net.decompose()
    with pytest.raises(ValueError, match="weighted"):
        net.compute_kconnectivity()

    multi = nx.MultiGraph(plain.edges())
    net = Network(multi, LaNetConfig(graph=GraphConfig(multigraph=True)))
    net.decompose()
    with pytest.raises(ValueError, match="multigraph"):
        net.compute_kconnectivity()

    digraph = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
    config = LaNetConfig(
        graph=GraphConfig(directed=True),
        decomposition=DecompositionConfig(decomp_type=DecompositionType.DCORES),
    )
    net = Network(digraph, config)
    net.decompose()
    with pytest.raises(ValueError, match="k-core"):
        net.compute_kconnectivity()

    # The configuration refuses the combinations up front
    with pytest.raises(ValueError, match="k-core"):
        DecompositionConfig(kconn=True, decomp_type=DecompositionType.KDENSES)
    for flag in ("weighted", "directed", "multigraph"):
        with pytest.raises(ValueError, match=flag):
            LaNetConfig(
                graph=GraphConfig(**{flag: True}),
                decomposition=DecompositionConfig(kconn=True),
            )
    assert DecompositionConfig(kconn=True).kconn_type == KConnectivityType.WIDE
