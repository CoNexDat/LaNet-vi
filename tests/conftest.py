"""Shared pytest fixtures."""

from pathlib import Path

import matplotlib
import networkx as nx
import pytest

matplotlib.use("Agg")


@pytest.fixture
def karate() -> nx.Graph:
    """Zachary's karate club graph (34 nodes, 78 edges)."""
    return nx.karate_club_graph()


@pytest.fixture
def small_edge_list(tmp_path: Path) -> Path:
    """Write a small undirected edge list (two triangles joined by a path) to a file."""
    path = tmp_path / "edges.txt"
    path.write_text("0 1\n1 2\n2 0\n2 3\n3 4\n4 5\n5 3\n5 6\n")
    return path
