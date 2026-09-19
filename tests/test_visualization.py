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
    """Every node gets a position, color and size."""
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
    """show_color_legend gates the color legend and show_degree_scale the degree legend."""
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
    """A node named "0" is labeled like any other (the old skip is gone)."""
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
    ("decomp_type", "measure", "title"),
    [
        (DecompositionType.KCORES, "mcore", "k-core"),
        (DecompositionType.KDENSES, "mcore", "m-core"),
        (DecompositionType.KDENSES, "kdense", "k-dense"),
        (DecompositionType.DCORES, "mcore", "d-core"),
    ],
)
def test_color_legend_title_follows_decomposition(
    karate: nx.Graph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decomp_type, measure, title
):
    """The color legend is titled after the decomposition (#10) and the k-dense measure."""
    from lanet_vi.models.config import DecompositionConfig

    texts = _spy_texts(monkeypatch)
    graph = karate.to_directed() if decomp_type is DecompositionType.DCORES else karate
    config = LaNetConfig(
        visualization=VisualizationConfig(width=300, height=300),
        decomposition=DecompositionConfig(measure=measure),
    )
    net = Network(graph, config)
    net.decompose(decomp_type)
    net.visualize(tmp_path / "out.png")

    assert title in texts


def _spy_texts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every string drawn with ``Axes.text``."""
    from matplotlib.axes import Axes

    drawn: list[str] = []
    original_text = Axes.text

    def spy(self, x, y, s, *args, **kwargs):  # noqa: ANN001, ANN202
        drawn.append(str(s))
        return original_text(self, x, y, s, *args, **kwargs)

    monkeypatch.setattr(Axes, "text", spy)
    return drawn


def test_kdense_legend_labels_mcore_as_k_minus_2(
    karate: nx.Graph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """With the default -measure mcore the k-dense legend counts from 0; with kdense from 2."""
    from lanet_vi.models.config import DecompositionConfig

    labels = {}
    for measure in ("mcore", "kdense"):
        texts = _spy_texts(monkeypatch)
        config = LaNetConfig(
            visualization=VisualizationConfig(width=300, height=300, show_degree_scale=False),
            decomposition=DecompositionConfig(measure=measure),
        )
        net = Network(karate, config)
        net.decompose(DecompositionType.KDENSES)
        net.visualize(tmp_path / f"{measure}.png")
        labels[measure] = [int(t) for t in texts if t.lstrip("-").isdigit()]
        max_index = net.decomposition.max_index

    assert labels["kdense"] == list(range(2, max_index + 1))
    assert labels["mcore"] == list(range(0, max_index - 1))


def test_color_legend_hidden_with_custom_colors(
    karate: nx.Graph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A colors file replaces the shell colors and hides the color legend (C++ -colorsFile)."""
    from lanet_vi.visualization import matplotlib_renderer as mr

    calls: list[str] = []
    monkeypatch.setattr(mr, "_draw_degree_scale", lambda *a, **k: calls.append("color"))
    net = Network(karate, _small_config())
    net.node_colors = {0: (0.1, 0.2, 0.3)}
    net.decompose()
    layout = net.compute_layout()
    net.visualize(tmp_path / "out.png", layout=layout)

    assert calls == []
    assert layout.node_colors[0] == (0.1, 0.2, 0.3)
    # Nodes absent from the file are white on the (default) black background
    assert layout.node_colors[1] == (1.0, 1.0, 1.0)

    # An empty colors file still means "custom colors": all default, legend hidden
    empty = tmp_path / "colors.txt"
    empty.write_text("")
    net = Network(karate, _small_config())
    net.load_node_colors(empty)
    net.decompose()
    layout = net.compute_layout()
    net.visualize(tmp_path / "empty.png", layout=layout)
    assert calls == []
    assert set(layout.node_colors.values()) == {(1.0, 1.0, 1.0)}


def test_png_is_exactly_width_by_height(karate: nx.Graph, tmp_path: Path):
    """The picture is the requested pixel size, whatever the aspect ratio (#24)."""
    from PIL import Image

    config = LaNetConfig(visualization=VisualizationConfig(width=320, height=100))
    net = Network(karate, config)
    net.decompose()
    out = tmp_path / "wide.png"
    net.visualize(out)
    with Image.open(out) as image:
        assert image.size == (320, 100)


def test_edge_colors_are_flipped_and_shaded(karate: nx.Graph):
    """The half next to u takes v's color darkened by 0.75 (col) or lightened by 1.2 (bw)."""
    from lanet_vi.models.config import ColorScheme
    from lanet_vi.visualization.colors import scale_color

    for scheme, factor in ((ColorScheme.COLOR, 0.75), (ColorScheme.GRAYSCALE, 1.2)):
        config = LaNetConfig(
            visualization=VisualizationConfig(
                width=300, height=300, color_scheme=scheme, edges_percent=1.0
            )
        )
        G = nx.Graph(karate.edges())
        net = Network(G, config)
        net.decompose()
        layout = net.compute_layout()
        assert layout.visible_edges == list(G.edges())
        for u, v in layout.visible_edges:
            near_u, near_v = layout.edge_colors[(u, v)]
            assert near_u == scale_color(layout.node_colors[v], factor)
            assert near_v == scale_color(layout.node_colors[u], factor)


def test_edge_width_follows_the_smaller_endpoint_degree(karate: nx.Graph):
    """Edge width is 0.2 host radii of min(degree) (2 x the C++ ratioEdge), in layout units."""
    from lanet_vi.visualization.lanet_layout import node_radius

    G = nx.Graph(karate.edges())
    config = LaNetConfig(visualization=VisualizationConfig(width=300, height=300, edges_percent=1))
    net = Network(G, config)
    net.decompose()
    layout = net.compute_layout()
    max_degree = max(d for _, d in G.degree())
    for u, v in layout.visible_edges:
        expected = 0.2 * node_radius(min(G.degree(u), G.degree(v)), max_degree)
        assert layout.edge_widths[(u, v)] == pytest.approx(expected)


def test_kdense_edges_take_their_own_index_color(karate: nx.Graph):
    """K-dense edges are one color: the shell color of the edge index, darkened by 0.5."""
    from lanet_vi.visualization.colors import compute_shell_color, scale_color

    config = LaNetConfig(visualization=VisualizationConfig(width=300, height=300, edges_percent=1))
    net = Network(karate, config)
    net.decompose(DecompositionType.KDENSES)
    layout = net.compute_layout()
    edge_indices = net.decomposition.metadata["edge_indices"]
    for u, v in layout.visible_edges:
        index = edge_indices[(min(u, v), max(u, v))]
        expected = scale_color(
            compute_shell_color(index, net.decomposition.max_index, "col", dense=True), 0.5
        )
        assert layout.edge_colors[(u, v)] == (expected, expected)


def test_mcore_color_scale_max_is_two_below_the_dense_index(karate: nx.Graph):
    """--color-scale-max 3 with mcore colors like --color-scale-max 5 with kdense."""
    from lanet_vi.models.config import DecompositionConfig

    colors = {}
    for measure, scale_max in (("mcore", 3), ("kdense", 5)):
        config = LaNetConfig(
            visualization=VisualizationConfig(
                width=300, height=300, color_scale_max_value=scale_max
            ),
            decomposition=DecompositionConfig(measure=measure),
        )
        net = Network(karate, config)
        net.decompose(DecompositionType.KDENSES)
        colors[measure] = net.compute_layout().node_colors
    assert colors["mcore"] == colors["kdense"]


def test_degree_legend_samples_match_the_drawn_radii(karate: nx.Graph):
    """The degree legend shows ceil(dmax / 4^i) with the radius those nodes have (#24)."""
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lanet_vi.visualization import matplotlib_renderer as mr
    from lanet_vi.visualization.lanet_layout import node_radius

    G = nx.Graph(karate.edges())
    config = LaNetConfig(
        visualization=VisualizationConfig(width=300, height=300, node_size_scale=2)
    )
    net = Network(G, config)
    net.decompose()
    layout = net.compute_layout()

    fig, ax = plt.subplots()
    mr._draw_size_legend(ax, G, config.visualization, layout, px_per_unit=1000.0)
    plt.close(fig)
    max_degree = max(d for _, d in G.degree())  # 17: samples 17, 5, 2
    expected = [2 * node_radius(math.ceil(max_degree / 4**i), max_degree) for i in range(3)]
    assert [p.radius for p in ax.patches] == pytest.approx(expected)
    assert [t.get_text() for t in ax.texts] == ["17", "5", "2", "degree"]


def test_degree_legend_shows_strengths_for_weighted_layouts(karate: nx.Graph):
    """Weighted layouts list strengths smax / 4^i; tiny weights fall back to degrees."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lanet_vi.models.config import GraphConfig
    from lanet_vi.visualization import matplotlib_renderer as mr
    from lanet_vi.visualization.lanet_layout import node_radius

    config = LaNetConfig(
        visualization=VisualizationConfig(width=300, height=300),
        graph=GraphConfig(weighted=True),
    )
    net = Network(karate, config)  # networkx's karate club carries weight attributes
    net.decompose()
    layout = net.compute_layout()
    assert layout.weighted
    strengths = {v: sum(d["weight"] for d in karate[v].values()) for v in karate}
    smax = max(strengths.values())
    dmax = max(d for _, d in karate.degree())

    fig, ax = plt.subplots()
    mr._draw_size_legend(ax, karate, config.visualization, layout, px_per_unit=1000.0)
    plt.close(fig)
    expected = [node_radius(0, dmax, smax / 4**i, smax, weighted=True) for i in range(5)]
    assert [p.radius for p in ax.patches] == pytest.approx(expected)
    labels = [t.get_text() for t in ax.texts]
    assert labels[-1] == "strength" and labels[0] == f"{smax:g}"

    # All strengths <= 1: the radii use the degree law and so does the legend
    tiny = nx.Graph(karate.edges())
    nx.set_edge_attributes(tiny, 0.01, "weight")
    net = Network(tiny, config)
    net.decompose()
    layout = net.compute_layout()
    assert layout.node_sizes[0] == pytest.approx(node_radius(karate.degree(0), dmax))
    assert len(set(layout.node_sizes.values())) > 1
    fig, ax = plt.subplots()
    mr._draw_size_legend(ax, tiny, config.visualization, layout, px_per_unit=1000.0)
    plt.close(fig)
    assert [t.get_text() for t in ax.texts] == ["17", "5", "2", "degree"]


def test_color_legend_has_one_circle_per_index_and_sparse_labels():
    """One circle per index from 1 to max; labels every max // 15 + 1 counted from the top."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lanet_vi.models.config import MeasureType
    from lanet_vi.models.graph import DecompositionResult
    from lanet_vi.visualization import matplotlib_renderer as mr

    decomposition = DecompositionResult(
        decomp_type="kcores", node_indices={i: i for i in range(1, 41)}, max_index=40, min_index=1
    )
    fig, ax = plt.subplots()
    mr._draw_degree_scale(ax, decomposition, VisualizationConfig(), 10.0, 1.0, MeasureType.MCORE)
    plt.close(fig)
    assert len(ax.patches) == 40
    labels = [t.get_text() for t in ax.texts]
    # 40 // 15 + 1 = 3: 40, 37, ..., 1 labeled, then the title
    assert labels == [str(i) for i in range(1, 41) if (40 - i) % 3 == 0] + ["k-core"]
    ys = sorted(p.center[1] for p in ax.patches)
    assert ys[0] == pytest.approx(-0.9 * 10.0 + 5.0 * 1.5 * 0.8 * (10.0 / 1.5) * 2.0 / (40 * 5))


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
    # The C++ viewport: 1.6 x 1.2 times the network radius (gamma * u * R)
    assert layout.bounds[1] == pytest.approx(1.6 * radii[0])
    assert layout.bounds[3] == pytest.approx(1.2 * radii[0])


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


def test_single_node_and_edgeless_graphs_render(tmp_path: Path):
    """The whole pipeline (layout, legends, PNG) survives graphs without edges."""
    for G in (nx.Graph([(0, 0)]), nx.empty_graph(5)):
        net = Network(G, _small_config())
        net.decompose()
        out = tmp_path / "edgeless.png"
        net.visualize(out)
        assert out.exists() and out.stat().st_size > 0


def test_pow_mode_reaches_the_layout_and_the_node_radii(karate: nx.Graph, tmp_path: Path):
    """--coord-distribution pow: unit-disc frame, packing parameters passed, pow radii."""
    import math
    from unittest.mock import patch

    from lanet_vi.models.config import CoordDistributionAlgorithm, LayoutConfig
    from lanet_vi.visualization import lanet_layout

    seen: dict[str, object] = {}
    original = lanet_layout.compute_lanet_layout

    def spy(graph, node_index, params, *args, **kwargs):  # noqa: ANN001, ANN202
        seen["params"] = params
        return original(graph, node_index, params, *args, **kwargs)

    G = nx.Graph(karate.edges())
    config = _small_config()
    config.layout = LayoutConfig(
        coord_distribution=CoordDistributionAlgorithm.POWER, alpha=0.2, beta=2.0, seed=3
    )
    net = Network(G, config)
    net.decompose()
    with patch("lanet_vi.core.network.compute_lanet_layout", spy):
        layout = net.compute_layout()
    params = seen["params"]
    assert (params.coord_distribution, params.alpha, params.beta, params.dense) == (
        "pow",
        0.2,
        2.0,
        False,
    )
    assert params.ratio_constant is None
    vis = config.visualization
    assert layout.frame == pytest.approx(vis.gamma * vis.unit_length)
    assert layout.radius_law is not None and layout.radius_law.modern
    degree = dict(G.degree())
    for v in G:
        expected = vis.node_size_scale * 0.007 * math.log(1 + degree[v]) ** 1.5
        assert layout.node_sizes[v] == pytest.approx(expected)
    # Edge widths follow the same law (0.1 host radii of the smaller endpoint degree)
    u, v = layout.visible_edges[0]
    assert layout.edge_widths[(u, v)] == pytest.approx(
        2 * 0.1 * vis.node_size_scale * 0.007 * math.log(1 + min(degree[u], degree[v])) ** 1.5
    )
    out = tmp_path / "pow.png"
    net.visualize(out, layout)
    assert out.exists() and out.stat().st_size > 0


def test_kdense_pow_mode_uses_the_dense_variant_and_its_ratio_constant(karate: nx.Graph):
    """K-dense + pow: the kdenses_component.cpp variant, radii ratioConstant sqrt(log(1+d))."""
    import math

    from lanet_vi.models.config import (
        CoordDistributionAlgorithm,
        DecompositionConfig,
        DecompositionType,
        LayoutConfig,
    )

    G = nx.Graph(karate.edges())
    config = _small_config()
    config.decomposition = DecompositionConfig(decomp_type=DecompositionType.KDENSES)
    config.layout = LayoutConfig(coord_distribution=CoordDistributionAlgorithm.LOG)
    net = Network(G, config)
    net.decompose()
    layout = net.compute_layout()
    law = layout.radius_law
    assert law is not None and law.modern and law.dense
    assert 0 < law.ratio_constant < 1.0  # lowered by the top core
    degree = dict(G.degree())
    scale = config.visualization.node_size_scale
    for v in G:
        expected = scale * law.ratio_constant * math.sqrt(math.log(1 + degree[v]))
        assert layout.node_sizes[v] == pytest.approx(expected)
    # An explicit ratio constant is used as is
    config.layout = LayoutConfig(
        coord_distribution=CoordDistributionAlgorithm.LOG, ratio_constant=0.25
    )
    net = Network(G, config)
    net.decompose()
    assert net.compute_layout().radius_law.ratio_constant == 0.25


def test_degree_legend_rows_never_overlap_in_pow_mode(karate: nx.Graph):
    """Large pow-mode samples push the rows apart; the label clears the circle."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lanet_vi.models.config import CoordDistributionAlgorithm, LayoutConfig
    from lanet_vi.visualization import matplotlib_renderer as mr

    G = nx.Graph(karate.edges())
    config = _small_config()
    config.visualization.node_size_scale = 20  # samples far bigger than the C++ pitch
    config.layout = LayoutConfig(coord_distribution=CoordDistributionAlgorithm.POWER)
    net = Network(G, config)
    net.decompose()
    layout = net.compute_layout()

    fig, ax = plt.subplots()
    mr._draw_size_legend(ax, G, config.visualization, layout, px_per_unit=1000.0)
    plt.close(fig)
    circles = [(p.center[1], p.radius) for p in ax.patches]
    for (y_low, r_low), (y_high, r_high) in zip(circles, circles[1:], strict=False):
        assert y_high - y_low > r_low + r_high
    labels = [t for t in ax.texts if t.get_text() != "degree"]
    for (y, r), text in zip(circles, labels, strict=True):
        assert text.get_position()[0] > -layout.frame * 15.0 / 12.0 + r
        assert text.get_position()[1] == pytest.approx(y)
    title = next(t for t in ax.texts if t.get_text() == "degree")
    assert title.get_position()[1] > circles[-1][0] + circles[-1][1]


def test_window_crops_the_viewport_from_the_top_left(karate: nx.Graph):
    """--window hstart hend vstart vend: a fraction of the full 3.2 x 2.4 frame viewport."""
    G = nx.Graph(karate.edges())
    config = _small_config()
    net = Network(G, config)
    net.decompose()
    full = net.compute_layout()
    frame = full.frame
    assert full.bounds[0] <= -1.6 * frame and full.bounds[3] >= 1.2 * frame

    config.visualization.window = (0.0, 0.5, 0.0, 0.5)  # top-left quarter
    net = Network(G, config)
    net.decompose()
    quarter = net.compute_layout()
    assert quarter.bounds == pytest.approx((-1.6 * frame, 0.0, 0.0, 1.2 * frame))
    assert quarter.frame == full.frame  # the legends keep their place, cropped or not

    config.visualization.window = (0.25, 0.75, 0.5, 1.0)  # middle half, bottom half
    net = Network(G, config)
    net.decompose()
    bounds = net.compute_layout().bounds
    assert bounds == pytest.approx((-0.8 * frame, 0.8 * frame, -1.2 * frame, 0.0))


def test_window_must_be_a_sub_rectangle():
    """Start must be below end and both inside [0, 1]."""
    from pydantic import ValidationError

    for window in ((0.5, 0.4, 0.0, 1.0), (0.0, 1.0, 0.2, 0.2), (-0.1, 1.0, 0.0, 1.0)):
        with pytest.raises(ValidationError, match="window"):
            VisualizationConfig(window=window)
    assert VisualizationConfig(window=(0.1, 0.9, 0.2, 0.8)).window == (0.1, 0.9, 0.2, 0.8)
