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
