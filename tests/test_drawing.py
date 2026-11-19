"""Tests for the NetworkX-facing helpers: ``lanet_vi.draw`` and ``lanet_vi.lanetvi_layout``."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
from matplotlib.figure import Figure

import lanet_vi
from lanet_vi.core.network import Network
from lanet_vi.drawing import _integer_graph
from lanet_vi.models.config import DecompositionConfig, LaNetConfig, VisualizationConfig


def _small_config() -> LaNetConfig:
    return LaNetConfig(visualization=VisualizationConfig(width=300, height=300))


def _named_graph() -> nx.Graph:
    """Two triangles joined by a path, with string labels."""
    return nx.Graph(
        [("a", "b"), ("b", "c"), ("c", "a"), ("c", "d"), ("d", "e"), ("e", "f"), ("f", "d")]
    )


def test_exported_at_top_level():
    """Both helpers are importable from the package root."""
    assert lanet_vi.draw is lanet_vi.drawing.draw
    assert lanet_vi.lanetvi_layout is lanet_vi.drawing.lanetvi_layout


def test_integer_graph_keeps_integer_labels(karate: nx.Graph):
    """A graph with int labels is used as it is, with no mapping."""
    graph, labels = _integer_graph(karate)
    assert graph is karate
    assert labels is None


def test_integer_graph_relabels_other_labels():
    """String, tuple and bool labels are numbered 0..n-1 in node order."""
    graph = nx.Graph([("x", (1, 2)), ((1, 2), True)])
    relabeled, labels = _integer_graph(graph)
    assert labels == {0: "x", 1: (1, 2), 2: True}
    assert set(relabeled.nodes) == {0, 1, 2}
    assert relabeled.has_edge(0, 1) and relabeled.has_edge(1, 2)


def test_integer_graph_keeps_edge_data():
    """Relabeling keeps edge attributes (weights)."""
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=3.0)
    relabeled, _ = _integer_graph(graph)
    assert relabeled.edges[0, 1]["weight"] == 3.0


def test_layout_keys_are_the_original_nodes():
    """Positions are keyed by the graph's own labels, as NetworkX layouts are."""
    graph = _named_graph()
    pos = lanet_vi.lanetvi_layout(graph, _small_config())
    assert set(pos) == set(graph.nodes)
    for xy in pos.values():
        assert isinstance(xy, np.ndarray)
        assert xy.shape == (2,)


def test_layout_scale_and_center(karate: nx.Graph):
    """``scale`` and ``center`` behave as in ``nx.spring_layout``."""
    pos = lanet_vi.lanetvi_layout(karate, _small_config(), scale=2.0, center=(10.0, -5.0))
    coords = np.array(list(pos.values()))
    assert np.isclose(np.abs(coords - (10.0, -5.0)).max(), 2.0)
    assert np.allclose(coords.mean(axis=0), (10.0, -5.0))


def test_layout_without_scale_matches_network(karate: nx.Graph):
    """``scale=None`` returns LaNet-vi's own coordinates."""
    config = _small_config()
    pos = lanet_vi.lanetvi_layout(karate, config, scale=None)
    net = Network(karate, config)
    net.decompose()
    layout = net.compute_layout()
    for node, xy in layout.node_positions.items():
        assert np.allclose(pos[node], xy)


def test_layout_works_with_nx_draw(karate: nx.Graph):
    """The positions can be passed straight to ``nx.draw``."""
    fig, ax = plt.subplots()
    nx.draw(karate, lanet_vi.lanetvi_layout(karate, _small_config()), ax=ax)
    plt.close(fig)


def test_layout_directed_and_multigraph():
    """Directed graphs and multigraphs get a position for every node."""
    directed = nx.gnp_random_graph(20, 0.2, directed=True, seed=1)
    assert len(lanet_vi.lanetvi_layout(directed, _small_config())) == 20
    multi = nx.MultiGraph([("a", "b"), ("a", "b"), ("b", "c")])
    assert set(lanet_vi.lanetvi_layout(multi, _small_config())) == {"a", "b", "c"}


def test_layout_from_layer_drops_lower_nodes():
    """With ``from_layer`` only the nodes of the kept layers get a position."""
    config = _small_config()
    config.decomposition = DecompositionConfig(from_layer=2)
    graph = _named_graph()
    graph.add_edge("f", "leaf")
    pos = lanet_vi.lanetvi_layout(graph, config)
    assert "leaf" not in pos
    assert set(pos) == {"a", "b", "c", "d", "e", "f"}


def test_empty_graph_is_refused():
    """A graph with no nodes raises a clear error."""
    with pytest.raises(ValueError, match="no nodes"):
        lanet_vi.lanetvi_layout(nx.Graph())
    with pytest.raises(ValueError, match="no nodes"):
        lanet_vi.draw(nx.Graph())


def test_draw_returns_a_figure_of_the_configured_size():
    """Without axes, ``draw`` makes a figure of the configured pixel size."""
    fig = lanet_vi.draw(_named_graph(), config=_small_config())
    assert isinstance(fig, Figure)
    width, height = fig.get_size_inches() * fig.dpi
    assert (round(width), round(height)) == (300, 300)
    plt.close(fig)


def test_draw_on_given_axes():
    """With axes, ``draw`` draws on them and returns their figure."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(6, 3))
    returned = lanet_vi.draw(_named_graph(), ax=right, config=_small_config())
    assert returned is fig
    assert right.collections  # nodes and edges were drawn on the given axes
    assert not left.collections
    plt.close(fig)


def test_draw_labels_show_original_names():
    """Node labels in the picture are the graph's own labels."""
    config = _small_config()
    config.visualization.show_node_labels = True
    fig = lanet_vi.draw(_named_graph(), config=config)
    texts = {text.get_text() for text in fig.axes[0].texts}
    assert {"a", "b", "c", "d", "e", "f"} <= texts
    plt.close(fig)
