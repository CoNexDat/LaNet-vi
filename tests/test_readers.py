"""Tests for the edge-list, names and colours readers."""

from pathlib import Path

import networkx as nx
import pytest

from lanet_vi.decomposition.kcores import compute_kcores
from lanet_vi.io.readers import read_edge_list, read_node_colors, read_node_names
from lanet_vi.io.writers import write_edge_list


def _write(tmp_path: Path, text: str, name: str = "edges.txt") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_read_edge_list_accepts_tabs_and_repeated_spaces(tmp_path: Path):
    """Any run of whitespace separates columns, as in the C++ reader."""
    path = _write(tmp_path, "1\t2\n2   3\n 3 1 \n")

    G = read_edge_list(path)

    assert sorted(G.edges()) == [(1, 2), (1, 3), (2, 3)]


def test_read_edge_list_ignores_extra_column_when_unweighted(tmp_path: Path):
    """A third column is ignored when the graph is not weighted."""
    path = _write(tmp_path, "1 2 0.5\n2 3 1.5\n")

    G = read_edge_list(path)

    assert G.number_of_edges() == 2
    assert "weight" not in G[1][2]


def test_read_edge_list_weighted_without_weight_column_uses_one(tmp_path: Path):
    """A weighted read of a two-column file gets weight 1.0 (C++ behaviour)."""
    path = _write(tmp_path, "1 2\n2 3\n")

    G = read_edge_list(path, weighted=True)

    assert G[1][2]["weight"] == 1.0
    assert G[2][3]["weight"] == 1.0


def test_read_edge_list_drops_self_loops(tmp_path: Path):
    """Self-loops are dropped so the decompositions can run."""
    path = _write(tmp_path, "1 1\n1 2\n2 2\n2 3\n")

    G = read_edge_list(path)

    assert nx.number_of_selfloops(G) == 0
    assert G.number_of_edges() == 2


def test_read_edge_list_multigraph_keeps_parallel_edges(tmp_path: Path):
    """With multigraph=True parallel edges are preserved."""
    path = _write(tmp_path, "1 2\n1 2\n2 3\n")

    G = read_edge_list(path, multigraph=True)

    assert G.is_multigraph()
    assert G.number_of_edges() == 3
    assert G.number_of_edges(1, 2) == 2


def test_read_edge_list_rejects_non_integer_ids(tmp_path: Path):
    """Non-integer node ids raise a clear ValueError."""
    path = _write(tmp_path, "a b\n")

    with pytest.raises(ValueError, match="integer"):
        read_edge_list(path)


def test_read_edge_list_rejects_single_column(tmp_path: Path):
    """A file with one column is rejected."""
    path = _write(tmp_path, "1\n2\n")

    with pytest.raises(ValueError, match="two columns"):
        read_edge_list(path)


def test_edge_list_round_trip_with_default_delimiters(tmp_path: Path):
    """write_edge_list output is readable by read_edge_list with defaults."""
    G = nx.karate_club_graph()
    path = tmp_path / "out.txt"

    write_edge_list(G, path)
    back = read_edge_list(path)

    assert {frozenset(e) for e in back.edges()} == {frozenset(e) for e in G.edges()}


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


def test_read_node_names_keeps_spaces_and_strips_quotes(tmp_path: Path):
    """Names are the rest of the line; quotes are stripped; empty names allowed."""
    path = _write(tmp_path, '1 New York\n2 "Buenos Aires"\n# comment\n3\n', "names.txt")

    names = read_node_names(path)

    assert names == {1: "New York", 2: "Buenos Aires", 3: ""}


def test_read_node_colors_accepts_tabs(tmp_path: Path):
    """Colours files may be tab- or space-separated."""
    path = _write(tmp_path, "1\t1.0\t0.0\t0.0\n2 0 1 0\n", "colors.txt")

    colors = read_node_colors(path)

    assert colors == {1: (1.0, 0.0, 0.0), 2: (0.0, 1.0, 0.0)}


def test_read_edge_list_rejects_fractional_ids(tmp_path: Path):
    """Float-valued ids such as 1.5 are rejected rather than truncated."""
    path = _write(tmp_path, "1.5 2\n")

    with pytest.raises(ValueError, match="integer"):
        read_edge_list(path)


def test_read_edge_list_rejects_non_numeric_weight(tmp_path: Path):
    """A malformed weight is an error; only an absent weight defaults to 1.0."""
    path = _write(tmp_path, "1 2 not-a-number\n")

    with pytest.raises(ValueError, match="not a number"):
        read_edge_list(path, weighted=True)


def test_read_edge_list_short_row_weight_defaults_to_one(tmp_path: Path):
    """A row without a third field on a weighted graph gets weight 1.0."""
    path = _write(tmp_path, "1 2 0.5\n2 3\n")

    G = read_edge_list(path, weighted=True)

    assert G[1][2]["weight"] == 0.5
    assert G[2][3]["weight"] == 1.0


def test_read_edge_list_keeps_node_that_only_has_a_self_loop(tmp_path: Path):
    """Dropping a self-loop must not drop the node."""
    path = _write(tmp_path, "1 1\n2 3\n")

    G = read_edge_list(path)

    assert 1 in G and G.number_of_edges() == 1


def test_read_node_names_honours_explicit_delimiter(tmp_path: Path):
    """The delimiter argument still works for callers that pass one."""
    path = _write(tmp_path, "1,New York\n2,Paris\n", "names.csv")

    assert read_node_names(path, delimiter=",") == {1: "New York", 2: "Paris"}


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


def test_read_node_names_strips_trailing_comment(tmp_path: Path):
    """An inline # comment after the name is dropped, as pandas used to do."""
    path = _write(tmp_path, "1 Alice # note\n", "names.txt")

    assert read_node_names(path) == {1: "Alice"}


def test_weighted_digraph_strength_counts_both_directions():
    """Reciprocal weighted arcs are summed into one undirected weight."""
    from lanet_vi.decomposition.kcores import _as_weighted_simple_graph

    G = nx.DiGraph()
    G.add_weighted_edges_from([(1, 2, 1.0), (2, 1, 2.0)])

    merged = _as_weighted_simple_graph(G)
    assert not merged.is_directed()
    assert merged[1][2]["weight"] == 3.0


def test_read_edge_list_optional_weight_in_any_row_order(tmp_path: Path):
    """A short first row followed by a three-field row parses (fixed schema)."""
    path = _write(tmp_path, "1 2\n2 3 0.5\n")

    G = read_edge_list(path, weighted=True)

    assert G[1][2]["weight"] == 1.0 and G[2][3]["weight"] == 0.5


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
