"""Tests for decomposition export helpers."""

import json
from pathlib import Path

import networkx as nx
import pandas as pd

from lanet_vi.community.base import Community, CommunityResult
from lanet_vi.core.network import Network
from lanet_vi.decomposition.kcores import compute_kcores
from lanet_vi.io.writers import (
    write_community_json,
    write_decomposition_csv,
    write_decomposition_json,
    write_edge_list,
    write_graph_json,
    write_node_attributes,
)


def test_write_decomposition_csv_roundtrip(karate: nx.Graph, tmp_path: Path):
    """CSV export contains one row per node with the k-core index."""
    result = compute_kcores(karate)
    out = tmp_path / "cores.csv"

    write_decomposition_csv(result, out)

    df = pd.read_csv(out)
    assert list(df.columns) == ["node_id", "kcores_index"]
    assert len(df) == karate.number_of_nodes()
    assert dict(zip(df["node_id"], df["kcores_index"], strict=True)) == result.node_indices


def test_write_decomposition_json_includes_components(karate: nx.Graph, tmp_path: Path):
    """JSON export carries indices, components and statistics."""
    net = Network(karate)
    result = net.decompose()
    assert result.components, "Network.decompose() should populate components"
    out = tmp_path / "cores.json"

    write_decomposition_json(result, out)

    data = json.loads(out.read_text())
    assert data["decomposition_type"] == "kcores"
    assert data["num_nodes"] == karate.number_of_nodes()
    assert data["max_index"] == result.max_index
    assert len(data["node_indices"]) == karate.number_of_nodes()
    assert data["num_components"] == len(result.components)
    assert data["component_statistics"]["total_components"] == len(result.components)
    first = data["components"][0]
    assert set(first) == {"id", "index", "size", "nodes"}
    assert first["index"] is not None


def test_write_edge_list_roundtrip(karate: nx.Graph, tmp_path: Path):
    """Edge list export (with weights) can be read back into an identical graph."""
    out = tmp_path / "edges.txt"

    write_edge_list(karate, out)  # karate club edges carry a "weight" attribute

    G = nx.read_edgelist(out, nodetype=int, data=(("weight", float),))
    assert G.number_of_nodes() == karate.number_of_nodes()
    assert G.number_of_edges() == karate.number_of_edges()
    assert G[0][1]["weight"] == karate[0][1]["weight"]


def test_write_edge_list_without_weights(karate: nx.Graph, tmp_path: Path):
    """Weights can be dropped from the export."""
    out = tmp_path / "edges_plain.txt"

    write_edge_list(karate, out, include_weights=False, delimiter=" ")

    first_line = out.read_text().splitlines()[0]
    assert len(first_line.split()) == 2
    G = nx.read_edgelist(out, nodetype=int, data=False)
    assert G.number_of_edges() == karate.number_of_edges()


def test_write_node_attributes_csv(tmp_path: Path):
    """One row per node, ``node_id`` index, one column per attribute."""
    out = tmp_path / "attrs.csv"
    write_node_attributes({1: {"degree": 3, "shell": 2}, 2: {"degree": 1, "shell": 1}}, out)

    df = pd.read_csv(out)
    assert list(df.columns) == ["node_id", "degree", "shell"]
    assert df.set_index("node_id").loc[1, "shell"] == 2
    assert len(df) == 2


def test_write_graph_json_node_link_roundtrip(tmp_path: Path):
    """The node-link JSON carries the attributes and loads back into the same graph."""
    graph = nx.Graph()
    graph.add_node(0, name="a")
    graph.add_edge(0, 1, weight=2.5)
    out = tmp_path / "graph.json"

    write_graph_json(graph, out)

    data = json.loads(out.read_text())
    assert {node["id"] for node in data["nodes"]} == {0, 1}
    assert data["links"][0]["weight"] == 2.5
    assert next(node for node in data["nodes"] if node["id"] == 0)["name"] == "a"
    assert {(link["source"], link["target"]) for link in data["links"]} == set(graph.edges())


def test_write_graph_json_can_strip_attributes(tmp_path: Path):
    """With the flags off only ids, sources and targets remain."""
    graph = nx.Graph()
    graph.add_node(0, name="a")
    graph.add_edge(0, 1, weight=2.5)
    out = tmp_path / "graph.json"

    write_graph_json(graph, out, include_node_attrs=False, include_edge_attrs=False)

    data = json.loads(out.read_text())
    assert all(set(node) == {"id"} for node in data["nodes"])
    assert all(set(link) == {"source", "target"} for link in data["links"])


def test_write_community_json_includes_statistics(tmp_path: Path):
    """Communities, the node map (string keys) and size statistics are written."""
    result = CommunityResult(
        algorithm="test",
        communities=[Community(id=0, nodes=[1, 2, 3]), Community(id=1, nodes=[4])],
        node_to_community={1: 0, 2: 0, 3: 0, 4: 1},
        modularity=0.25,
    )
    out = tmp_path / "communities.json"

    write_community_json(result, out)

    data = json.loads(out.read_text())
    assert data["algorithm"] == "test"
    assert data["num_communities"] == 2
    assert data["modularity"] == 0.25
    assert data["communities"][0] == {"id": 0, "size": 3, "nodes": [1, 2, 3]}
    assert data["node_to_community"] == {"1": 0, "2": 0, "3": 0, "4": 1}
    assert data["statistics"] == {
        "largest_community": 3,
        "smallest_community": 1,
        "mean_community_size": 2.0,
    }


def test_write_community_json_with_no_communities(tmp_path: Path):
    """An empty result writes zero statistics instead of failing on ``max([])``."""
    result = CommunityResult(algorithm="test", communities=[], node_to_community={})
    out = tmp_path / "empty.json"

    write_community_json(result, out)

    data = json.loads(out.read_text())
    assert data["statistics"] == {
        "largest_community": 0,
        "smallest_community": 0,
        "mean_community_size": 0,
    }
