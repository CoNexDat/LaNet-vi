"""Tests for k-core decomposition."""

from pathlib import Path

import networkx as nx

from lanet_vi.decomposition.kcores import compute_kcores
from lanet_vi.io.readers import read_edge_list
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


# --- multigraph, self-loop and weighted-detection behaviour (#22)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "edges.txt"
    path.write_text(text)
    return path


def test_multigraph_kcores_count_parallel_edges(tmp_path: Path):
    """Two parallel edges between 1 and 2 give them degree 2, so they form a 2-core."""
    path = _write(tmp_path, "1 2\n1 2\n2 3\n")
    G = read_edge_list(path, multigraph=True)

    result = compute_kcores(G)

    assert result.node_indices == {1: 2, 2: 2, 3: 1}


def test_multigraph_kcores_match_simple_graph_when_no_parallel_edges():
    """A MultiGraph without parallel edges gives the same core numbers as nx.Graph."""
    simple = nx.karate_club_graph()
    multi = nx.MultiGraph(simple)

    assert compute_kcores(multi).node_indices == compute_kcores(simple).node_indices


def test_kcores_ignore_self_loops():
    """Self-loops do not change core numbers and do not crash."""
    G = nx.karate_club_graph()
    expected = compute_kcores(G).node_indices
    G.add_edge(0, 0)

    assert compute_kcores(G).node_indices == expected


def test_kcores_weighted_graph_with_only_self_loops():
    """A weighted graph whose only edges are self-loops does not crash."""
    G = nx.Graph()
    G.add_edge(1, 1, weight=2.0)
    G.add_node(2)

    result = compute_kcores(G)

    assert result.node_indices == {1: 0, 2: 0}


def test_multidigraph_kcores_count_both_directions():
    """For a directed multigraph, removing a node decrements predecessors too."""
    x, y, u, v = 0, 1, 2, 3
    G = nx.MultiDiGraph()
    G.add_edges_from([(x, y), (y, x), (x, u), (u, v)])

    result = compute_kcores(G)  # type: ignore[arg-type]

    assert result.node_indices == {x: 2, y: 2, u: 1, v: 1}


def test_weighted_multigraph_sums_parallel_weights():
    """Parallel weighted edges are merged by summing before the strength-based cores."""
    from lanet_vi.decomposition.kcores import _as_weighted_simple_graph

    G = nx.MultiGraph()
    G.add_weighted_edges_from([(1, 2, 0.5), (1, 2, 1.5), (2, 3, 1.0)])

    merged = _as_weighted_simple_graph(G)
    assert not merged.is_multigraph()
    assert merged[1][2]["weight"] == 2.0

    result = compute_kcores(G)  # runs the weighted path through the merge
    assert result.p_function is not None
    assert set(result.node_indices) == {1, 2, 3}


def test_weighted_flag_wins_over_late_weight_attribute():
    """weighted=True forces the strength path even if the first edges carry no weight."""
    G = nx.path_graph(150)
    G.add_edge(0, 149, weight=3.0)

    assert compute_kcores(G, weighted=False).p_function is None
    assert compute_kcores(G).p_function is not None  # autodetect scans every edge
    assert compute_kcores(G, weighted=True).p_function is not None


def test_weighted_digraph_strength_counts_both_directions():
    """Reciprocal weighted arcs are summed into one undirected weight."""
    from lanet_vi.decomposition.kcores import _as_weighted_simple_graph

    G = nx.DiGraph()
    G.add_weighted_edges_from([(1, 2, 1.0), (2, 1, 2.0)])

    merged = _as_weighted_simple_graph(G)
    assert not merged.is_directed()
    assert merged[1][2]["weight"] == 3.0


def test_kcores_weighted_flag_on_edgeless_graph():
    """weighted=True on a graph without edges returns zero cores instead of crashing."""
    G = nx.Graph()
    G.add_nodes_from([1, 2])

    assert compute_kcores(G, weighted=True).node_indices == {1: 0, 2: 0}


def test_network_directed_multigraph_kcores_end_to_end(tmp_path: Path):
    """--directed --multigraph works through Network.decompose (weak components)."""
    from lanet_vi.core.network import Network
    from lanet_vi.models.config import DecompositionType, GraphConfig, LaNetConfig

    path = _write(tmp_path, "1 2\n1 2\n2 3\n3 1\n")
    net = Network.from_edge_list(
        path, LaNetConfig(graph=GraphConfig(directed=True, multigraph=True))
    )

    result = net.decompose(DecompositionType.KCORES)

    assert result.components
    assert sum(c.size for c in result.components) == 3


def test_network_autodetects_weights_without_config():
    """Network(G) with weighted edges still takes the strength path by default."""
    from lanet_vi.core.network import Network

    G = nx.karate_club_graph()
    nx.set_edge_attributes(G, 1.0, "weight")

    assert Network(G).decompose().p_function is not None
