"""Color scale of the C++ LaNet-vi (types.cpp, computeHostColorByShellIndex)."""

import pytest

from lanet_vi.models.config import BackgroundColor, ColorScheme
from lanet_vi.visualization.colors import (
    BLACKWHITE,
    RAINBOW,
    clamp_color,
    compute_shell_color,
    create_matplotlib_colormap,
    default_node_color,
    get_color_scale,
    scale_color,
)


def _luminance(color: tuple[float, float, float]) -> float:
    return sum(color) / 3.0


def test_rainbow_runs_from_magenta_to_red():
    """The lowest shell is magenta and the top shell red, as in the C++ (#24)."""
    low = compute_shell_color(1, 10, ColorScheme.COLOR)
    top = compute_shell_color(10, 10, ColorScheme.COLOR)
    assert low == pytest.approx((1.0, 0.24, 1.0))  # magenta, luminosity 1.2 (odd), clamped
    assert top == pytest.approx((0.7, 0.14, 0.14))  # red, luminosity 0.7 (even)
    assert compute_shell_color(11, 11, ColorScheme.COLOR) == pytest.approx((1.0, 0.24, 0.24))


def test_luminosity_alternates_between_consecutive_shells():
    """Odd shells get 1.2 (clamped), even shells 0.7 of the scale color."""
    colors = [compute_shell_color(i, 20, ColorScheme.COLOR) for i in range(1, 21)]
    for i, color in enumerate(colors, start=1):
        base = compute_shell_color(i, 20, ColorScheme.GRAYSCALE)  # any: check the ratio instead
        assert base is not None
        if i % 2 == 0:
            assert max(color) <= 0.7 + 1e-9
        else:
            assert max(color) == pytest.approx(1.0)


def test_grayscale_runs_light_to_dark():
    """bw: white at the periphery, near black at the maximum (the Python used to invert it)."""
    colors = [compute_shell_color(i, 8, ColorScheme.GRAYSCALE) for i in range(1, 9)]
    assert colors[0] == (1.0, 1.0, 1.0)
    assert colors[-1] == pytest.approx((0.196, 0.196, 0.196), abs=1e-3)
    assert all(_luminance(a) > _luminance(b) for a, b in zip(colors, colors[1:], strict=False))
    assert all(color[0] == color[1] == color[2] for color in colors)


def test_interlaced_grayscale_alternates_halves_of_the_scale():
    """bwi: indices of the parity of max use the dark half, the others the light half."""
    colors = [compute_shell_color(i, 8, ColorScheme.GRAYSCALE_INTERLACED) for i in range(1, 9)]
    same_parity = [_luminance(c) for i, c in enumerate(colors, start=1) if (i + 8) % 2 == 0]
    other_parity = [_luminance(c) for i, c in enumerate(colors, start=1) if (i + 8) % 2 == 1]
    assert max(same_parity) < min(other_parity)
    assert colors[-1] == pytest.approx((0.196, 0.196, 0.196), abs=1e-3)  # position 1: as bw


def test_single_shell_is_red_and_scale_max_clamps():
    """A single shell is red; indices above color_scale_max share its color."""
    assert compute_shell_color(1, 1, ColorScheme.COLOR) == pytest.approx((1.0, 0.24, 0.24))
    capped = compute_shell_color(9, 20, ColorScheme.COLOR, color_scale_max=5)
    assert capped == compute_shell_color(5, 20, ColorScheme.COLOR, color_scale_max=5)


def test_kdense_white_background_uses_constant_luminosity():
    """graphics_kdenses.cpp: 0.9 on white, the k-core alternation on black."""
    on_white = compute_shell_color(
        4, 6, ColorScheme.COLOR, background=BackgroundColor.WHITE, dense=True
    )
    on_black = compute_shell_color(
        4, 6, ColorScheme.COLOR, background=BackgroundColor.BLACK, dense=True
    )
    base = compute_shell_color(4, 6, ColorScheme.COLOR, background=BackgroundColor.WHITE)
    assert on_black == base  # black background: same as k-cores (even shell: 0.7)
    assert on_white == pytest.approx(scale_color(clamp_color(base), 0.9 / 0.7), abs=1e-9)


def test_every_channel_is_clamped():
    """Stops above 1 (white 2.0, luminosity 1.2) never leak out."""
    for scheme in ColorScheme:
        for i in range(1, 30):
            color = compute_shell_color(i, 29, scheme)
            assert all(0.0 <= c <= 1.0 for c in color)
    assert scale_color((0.9, 0.5, 0.1), 1.2) == pytest.approx((1.0, 0.6, 0.12))


def test_scales_and_defaults():
    """The bw and bwi schemes share the blackwhite stops; colors-file defaults are white/black."""
    assert get_color_scale(ColorScheme.COLOR) == RAINBOW
    assert get_color_scale(ColorScheme.GRAYSCALE) == BLACKWHITE
    assert get_color_scale(ColorScheme.GRAYSCALE_INTERLACED) == BLACKWHITE
    assert default_node_color(BackgroundColor.BLACK) == (1.0, 1.0, 1.0)
    assert default_node_color(BackgroundColor.WHITE) == (0.0, 0.0, 0.0)
    cmap = create_matplotlib_colormap(ColorScheme.COLOR)
    assert cmap(0.0)[:3] == pytest.approx((1.0, 0.2, 1.0), abs=0.02)
    assert cmap(1.0)[:3] == pytest.approx((1.0, 0.2, 0.2), abs=0.02)


def test_edge_sampling_is_a_seeded_bernoulli():
    """select_visible_edges keeps each edge with p = max(percent, min_edges / E), seeded."""
    import networkx as nx

    from lanet_vi.models.config import VisualizationConfig
    from lanet_vi.visualization.matplotlib_renderer import select_visible_edges

    G = nx.gnm_random_graph(200, 2000, seed=1)
    config = VisualizationConfig(edges_percent=0.3, min_edges=0)
    first = select_visible_edges(G, config, seed=7)
    assert first == select_visible_edges(G, config, seed=7)
    assert first != select_visible_edges(G, config, seed=8)
    assert 0.25 * 2000 < len(first) < 0.35 * 2000
    # The min_edges floor raises the probability, edges_percent >= 1 keeps everything
    floor = select_visible_edges(G, VisualizationConfig(edges_percent=0.0, min_edges=1000), seed=7)
    assert 0.45 * 2000 < len(floor) < 0.55 * 2000
    assert select_visible_edges(G, VisualizationConfig(edges_percent=1.0), seed=0) == list(
        G.edges()
    )
    assert select_visible_edges(G, VisualizationConfig(edges_percent=0.0, min_edges=0)) == []
