"""Tests for the edge-list, names and colours readers."""

from pathlib import Path

import networkx as nx
import pytest

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


def test_read_edge_list_rejects_row_missing_target(tmp_path: Path):
    """A single short row in an otherwise valid file is reported with its line number."""
    path = _write(tmp_path, "1 2\n3\n")

    with pytest.raises(ValueError, match="row 2 has fewer than two columns"):
        read_edge_list(path)


def test_edge_list_round_trip_with_default_delimiters(tmp_path: Path):
    """write_edge_list output is readable by read_edge_list with defaults."""
    G = nx.karate_club_graph()
    path = tmp_path / "out.txt"

    write_edge_list(G, path)
    back = read_edge_list(path)

    assert {frozenset(e) for e in back.edges()} == {frozenset(e) for e in G.edges()}


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


def test_read_node_names_strips_trailing_comment(tmp_path: Path):
    """An inline # comment after the name is dropped, as pandas used to do."""
    path = _write(tmp_path, "1 Alice # note\n", "names.txt")

    assert read_node_names(path) == {1: "Alice"}


def test_read_edge_list_optional_weight_in_any_row_order(tmp_path: Path):
    """A short first row followed by a three-field row parses (fixed schema)."""
    path = _write(tmp_path, "1 2\n2 3 0.5\n")

    G = read_edge_list(path, weighted=True)

    assert G[1][2]["weight"] == 1.0 and G[2][3]["weight"] == 0.5


@pytest.mark.parametrize("token", ["NA", "N/A", "NaN", "nan", "null"])
def test_read_edge_list_rejects_na_like_weights(tmp_path: Path, token: str):
    """NA-like tokens in the weight column are errors, not missing values."""
    path = _write(tmp_path, f"1 2 {token}\n2 3 0.5\n")

    with pytest.raises(ValueError, match="not a number"):
        read_edge_list(path, weighted=True)


def test_read_edge_list_rejects_na_like_ids(tmp_path: Path):
    """NA-like tokens in an id column are rejected as non-integer ids."""
    path = _write(tmp_path, "NA 2\n")

    with pytest.raises(ValueError, match="integer"):
        read_edge_list(path)


def test_read_edge_list_empty_file_gives_empty_graph(tmp_path: Path):
    """An empty file (e.g. generate with p=0) yields an empty graph, not a pandas error."""
    path = _write(tmp_path, "")

    G = read_edge_list(path)

    assert G.number_of_nodes() == 0 and G.number_of_edges() == 0
