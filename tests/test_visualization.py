"""End-to-end rendering tests on a small graph."""

from pathlib import Path

import networkx as nx
import pytest

from lanet_vi.core.network import Network
from lanet_vi.models.config import (
    DecompositionType,
    LaNetConfig,
    VisualizationConfig,
)


def _small_config() -> LaNetConfig:
    return LaNetConfig(visualization=VisualizationConfig(width=300, height=300))


def test_visualize_kcores_writes_png(karate: nx.Graph, tmp_path: Path):
    """K-core pipeline renders a non-empty PNG."""
    net = Network(karate, _small_config())
    net.decompose(DecompositionType.KCORES)
    out = tmp_path / "kcores.png"
    net.visualize(out)
    assert out.exists() and out.stat().st_size > 0


def test_visualize_kdenses_writes_png(karate: nx.Graph, tmp_path: Path):
    """K-dense pipeline renders a non-empty PNG."""
    net = Network(karate, _small_config())
    net.decompose(DecompositionType.KDENSES)
    out = tmp_path / "kdenses.png"
    net.visualize(out)
    assert out.exists() and out.stat().st_size > 0


def test_visualize_with_precomputed_layout(karate: nx.Graph, tmp_path: Path):
    """A layout computed once can be passed to visualize()."""
    net = Network(karate, _small_config())
    net.decompose()
    layout = net.compute_layout()
    out = tmp_path / "layout.png"
    net.visualize(out, layout=layout)
    assert out.exists() and out.stat().st_size > 0


def test_community_module_imports_on_python39():
    """community.base must not use PEP 604 unions (SyntaxError on 3.9)."""
    from lanet_vi.community.base import CommunityResult

    result = CommunityResult(algorithm="t", communities=[], node_to_community={})
    assert result.get_node_community(1) is None


def test_layout_positions_cover_all_nodes(karate: nx.Graph):
    """Every node gets a position, colour and size."""
    net = Network(karate, _small_config())
    net.decompose()
    layout = net.compute_layout()
    assert set(layout.node_positions) == set(karate.nodes())
    assert set(layout.node_colors) == set(karate.nodes())
    assert set(layout.node_sizes) == set(karate.nodes())
    xmin, xmax, ymin, ymax = layout.bounds
    assert xmin <= xmax and ymin <= ymax


@pytest.mark.parametrize(
    ("color_legend", "degree_legend"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_legend_switches_gate_their_helpers(
    karate: nx.Graph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, color_legend, degree_legend
):
    """show_color_legend gates the colour legend and show_degree_scale the degree legend."""
    from lanet_vi.models.config import LaNetConfig, VisualizationConfig
    from lanet_vi.visualization import matplotlib_renderer as mr

    calls: list[str] = []
    monkeypatch.setattr(mr, "_draw_degree_scale", lambda *a, **k: calls.append("color"))
    monkeypatch.setattr(mr, "_draw_size_legend", lambda *a, **k: calls.append("degree"))

    config = LaNetConfig(
        visualization=VisualizationConfig(
            width=300,
            height=300,
            show_color_legend=color_legend,
            show_degree_scale=degree_legend,
        )
    )
    net = Network(karate, config)
    net.decompose()
    net.visualize(tmp_path / "out.png")

    assert ("color" in calls) is color_legend
    assert ("degree" in calls) is degree_legend


def test_labels_are_drawn_for_every_named_node_including_zero(
    karate: nx.Graph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A node named "0" is labelled like any other (the old skip is gone)."""
    from matplotlib.axes import Axes

    from lanet_vi.models.config import LaNetConfig, VisualizationConfig

    drawn: list[str] = []
    original_text = Axes.text

    def spy(self, x, y, s, *args, **kwargs):  # noqa: ANN001, ANN202
        drawn.append(str(s))
        return original_text(self, x, y, s, *args, **kwargs)

    monkeypatch.setattr(Axes, "text", spy)

    config = LaNetConfig(
        visualization=VisualizationConfig(width=300, height=300, show_node_labels=True)
    )
    net = Network(karate, config)
    net.node_names = {0: "0", 1: "one"}
    net.decompose()
    net.visualize(tmp_path / "out.png")

    assert "0" in drawn and "one" in drawn


@pytest.mark.parametrize(
    ("decomp_type", "title"),
    [
        (DecompositionType.KCORES, "k-core"),
        (DecompositionType.KDENSES, "k-dense"),
        (DecompositionType.DCORES, "d-core"),
    ],
)
def test_color_legend_title_follows_decomposition(
    karate: nx.Graph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decomp_type, title
):
    """The colour legend is titled after the decomposition, not always "k-core" (#10)."""
    from matplotlib.axes import Axes

    titles: list[str] = []
    original_legend = Axes.legend

    def spy(self, *args, **kwargs):  # noqa: ANN001, ANN202
        if "title" in kwargs:
            titles.append(kwargs["title"])
        return original_legend(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "legend", spy)

    graph = karate.to_directed() if decomp_type is DecompositionType.DCORES else karate
    net = Network(graph, _small_config())
    net.decompose(decomp_type)
    net.visualize(tmp_path / "out.png")

    assert title in titles


def test_layout_components_are_the_nested_components_with_circles(karate: nx.Graph):
    """Border circles come from the nested component tree, only for components with clusters."""
    from lanet_vi.models.config import LayoutConfig

    G = nx.Graph(karate.edges())  # drop networkx's weight attribute: plain k-cores
    config = LaNetConfig(
        visualization=VisualizationConfig(width=300, height=300, gamma=1.5),
        layout=LayoutConfig(min_component_size=1),
    )
    net = Network(G, config)
    net.decompose()
    layout = net.compute_layout()

    # One component per shell 1..4, concentric, radius = ratio * u * gamma, one unit apart
    assert [c.shell_index for c in layout.components] == [1, 2, 3, 4]
    assert all(c.center == (0.0, 0.0) for c in layout.components)
    radii = [c.radius for c in layout.components]
    assert all(a - b == pytest.approx(1.5) for a, b in zip(radii, radii[1:], strict=False))
    assert layout.bounds[1] >= max(x for x, _ in layout.node_positions.values())
    assert layout.bounds[1] == pytest.approx(radii[0])  # the C++ camera frame


def test_autodetected_weights_use_weighted_geometry(karate: nx.Graph):
    """A graph with weight attributes but no --weighted still gets strength-based layout."""
    from unittest.mock import patch

    from lanet_vi.visualization import lanet_layout

    seen: dict[str, object] = {}
    original = lanet_layout.compute_lanet_layout

    def spy(graph, node_index, params, *args, **kwargs):  # noqa: ANN001, ANN202
        seen["weighted"] = params.weighted
        return original(graph, node_index, params, *args, **kwargs)

    net = Network(karate, _small_config())  # networkx's karate club carries weights
    net.decompose()
    assert net.decomposition is not None and net.decomposition.p_function is not None
    with patch("lanet_vi.core.network.compute_lanet_layout", spy):
        net.compute_layout()
    assert seen["weighted"] is True
