"""Tests for decomposition export helpers."""

import json
from pathlib import Path

import networkx as nx
import pandas as pd

from lanet_vi.core.network import Network
from lanet_vi.decomposition.kcores import compute_kcores
from lanet_vi.io.writers import (
    write_decomposition_csv,
    write_decomposition_json,
    write_edge_list,
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
