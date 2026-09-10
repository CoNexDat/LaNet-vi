"""End-to-end rendering tests on a small graph."""

from pathlib import Path

import networkx as nx

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
