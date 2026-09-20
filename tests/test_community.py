"""Tests for community detection, its data models and the community overlays."""

import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
from networkx.algorithms import community as nx_community

from lanet_vi.community import detect_communities
from lanet_vi.community.base import Community, CommunityResult
from lanet_vi.community.louvain import (
    detect_communities_greedy_modularity,
    detect_communities_louvain,
)
from lanet_vi.models.config import CommunityConfig
from lanet_vi.visualization.community_viz import (
    assign_node_colors_by_community,
    draw_community_boundaries,
    draw_community_circles,
    get_community_colors,
)


def test_community_result_lookup_helpers():
    """get_community / get_node_community work and return None when missing."""
    result = CommunityResult(
        algorithm="test",
        communities=[Community(id=0, nodes=[1, 2]), Community(id=1, nodes=[3])],
        node_to_community={1: 0, 2: 0, 3: 1},
    )
    assert result.num_communities == 2
    assert result.get_community(1) is not None
    assert result.get_community(1).size == 1  # type: ignore[union-attr]
    assert result.get_community(99) is None
    assert result.get_node_community(2) == 0
    assert result.get_node_community(42) is None
    assert result.get_community_sizes() == {0: 2, 1: 1}


def test_louvain_on_karate(karate: nx.Graph):
    """Louvain partitions every node of the karate club into a community."""
    result = detect_communities_louvain(karate, seed=0)
    assert set(result.node_to_community) == set(karate.nodes())
    assert result.num_communities >= 2


def test_get_community_colors_are_rgb_triples():
    """Community colors are RGB triples in [0, 1], including beyond 20 communities."""
    for n in (5, 25, 60):
        colors = get_community_colors(n)
        assert len(colors) == n
        for color in colors:
            assert len(color) == 3
            assert all(0.0 <= c <= 1.0 for c in color)
    assert get_community_colors(0) == []


def test_get_community_colors_are_distinct_and_honor_the_colormap():
    """Qualitative maps are used entry by entry (no two communities alike); others sampled."""
    cmap = plt.get_cmap("tab10")
    # Up to 10 communities: the dark tab10 shades, not tab20's dark/light pairs
    assert get_community_colors(4) == [tuple(float(c) for c in cmap(i)[:3]) for i in range(4)]
    for n in (10, 20, 25, 60):
        assert len(set(get_community_colors(n))) == n
    # A qualitative map given explicitly is honored entry by entry
    set3 = plt.get_cmap("Set3")
    assert get_community_colors(3, colormap="Set3") == [
        tuple(float(c) for c in set3(i)[:3]) for i in range(3)
    ]
    # A continuous map is sampled evenly over its range
    viridis = plt.get_cmap("viridis")
    assert get_community_colors(4, colormap="viridis") == [
        tuple(float(c) for c in viridis(i / 4)[:3]) for i in range(4)
    ]
    with pytest.raises(ValueError):
        get_community_colors(3, colormap="no-such-colormap")


def test_detect_communities_dispatches_on_the_config(karate: nx.Graph):
    """detect_communities runs the configured algorithm with its resolution and the seed."""
    louvain = detect_communities(karate, CommunityConfig(algorithm="louvain"), seed=1)
    assert louvain.algorithm == "louvain"
    assert louvain.node_to_community == detect_communities_louvain(karate, seed=1).node_to_community

    greedy = detect_communities(karate, CommunityConfig(algorithm="greedy_modularity"))
    assert greedy.algorithm == "greedy_modularity"
    assert (
        greedy.node_to_community == detect_communities_greedy_modularity(karate).node_to_community
    )

    # A higher resolution gives at least as many communities, for both algorithms
    for algorithm in ("louvain", "greedy_modularity"):
        low = detect_communities(karate, CommunityConfig(algorithm=algorithm, resolution=0.5))
        high = detect_communities(karate, CommunityConfig(algorithm=algorithm, resolution=2.0))
        assert high.num_communities >= low.num_communities

    config = CommunityConfig.model_construct(algorithm="bogus")
    with pytest.raises(ValueError, match="Unknown community algorithm"):
        detect_communities(karate, config)


def test_louvain_result_is_consistent_and_reproducible(karate: nx.Graph):
    """Communities partition the nodes, sizes match, modularity is NetworkX's, seed fixes it."""
    result = detect_communities_louvain(karate, seed=1)
    assert result.algorithm == "louvain"
    nodes = [node for community in result.communities for node in community.nodes]
    assert sorted(nodes) == sorted(karate.nodes())
    assert all(community.size == len(community.nodes) for community in result.communities)
    assert result.num_communities == len(result.communities)
    assert all(
        result.node_to_community[node] == community.id
        for community in result.communities
        for node in community.nodes
    )
    expected = nx_community.modularity(
        karate, [set(c.nodes) for c in result.communities], weight="weight"
    )
    assert result.modularity == pytest.approx(expected)
    assert 0.3 < result.modularity < 0.5  # the karate club's well-known range
    again = detect_communities_louvain(karate, seed=1)
    assert again.node_to_community == result.node_to_community


def test_louvain_falls_back_to_unweighted_when_no_weights(karate: nx.Graph):
    """Without a weight attribute the modularity is the unweighted one."""
    plain = nx.Graph(karate.edges())
    result = detect_communities_louvain(plain, seed=1)
    expected = nx_community.modularity(plain, [set(c.nodes) for c in result.communities])
    assert result.modularity == pytest.approx(expected)


def test_louvain_converts_directed_graphs():
    """A DiGraph is analyzed as its undirected version."""
    digraph = nx.DiGraph([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)])
    result = detect_communities_louvain(digraph, seed=0)
    assert set(result.node_to_community) == set(digraph.nodes())
    assert result.num_communities == 2


def test_greedy_modularity_on_two_triangles():
    """Greedy modularity separates two triangles joined by one edge."""
    graph = nx.Graph([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)])
    result = detect_communities_greedy_modularity(graph)
    assert result.algorithm == "greedy_modularity"
    assert result.num_communities == 2
    assert result.get_node_community(0) == result.get_node_community(1)
    assert result.get_node_community(0) != result.get_node_community(4)
    assert result.modularity == pytest.approx(
        nx_community.modularity(graph, [set(c.nodes) for c in result.communities])
    )
    assert set(result.node_to_community) == set(
        detect_communities_greedy_modularity(graph.to_directed()).node_to_community
    )


def _two_triangles_result() -> tuple[CommunityResult, dict[int, tuple[float, float]]]:
    """Two three-node communities plus a lone node, with hand-placed positions."""
    result = CommunityResult(
        algorithm="test",
        communities=[
            Community(id=0, nodes=[0, 1, 2]),
            Community(id=1, nodes=[3, 4, 5]),
            Community(id=2, nodes=[6]),
        ],
        node_to_community={0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2},
    )
    positions = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (0.0, 1.0),
        3: (5.0, 5.0),
        4: (6.0, 5.0),
        5: (5.0, 6.0),
        6: (9.0, 9.0),
    }
    return result, positions


def test_assign_node_colors_by_community_uses_palette_and_gray_fallback():
    """Nodes of one community share a color; a node without a community is gray."""
    result, positions = _two_triangles_result()
    positions[7] = (0.0, 9.0)  # positioned but in no community
    colors = assign_node_colors_by_community(result, positions)
    assert set(colors) == set(positions)
    palette = get_community_colors(result.num_communities)
    assert colors[0] == colors[1] == colors[2] == palette[0]
    assert colors[3] == palette[1]
    assert colors[0] != colors[3]
    assert colors[7] == (0.7, 0.7, 0.7)


def test_draw_community_boundaries_adds_one_hull_per_large_community():
    """Convex hulls are drawn for communities with at least three placed nodes only."""
    result, positions = _two_triangles_result()
    fig, ax = plt.subplots()
    try:
        draw_community_boundaries(ax, result, positions)
        assert len(ax.collections) == 1
        assert len(ax.collections[0].get_paths()) == 2  # the singleton is skipped
    finally:
        plt.close(fig)


def test_draw_community_boundaries_handles_degenerate_hulls(caplog: pytest.LogCaptureFixture):
    """Collinear points get a sliver hull; fewer than three distinct points get none."""
    result = CommunityResult(
        algorithm="test",
        communities=[Community(id=0, nodes=[0, 1, 2]), Community(id=1, nodes=[3, 4, 5])],
        node_to_community={0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1},
    )
    # Community 0 is collinear; community 1 has three nodes on two distinct positions
    # (nodes of one cluster can share a position in the layout)
    positions = {
        0: (0.0, 0.0),
        1: (1.0, 1.0),
        2: (2.0, 2.0),
        3: (5.0, 5.0),
        4: (5.0, 5.0),
        5: (6.0, 5.0),
    }
    fig, ax = plt.subplots()
    try:
        with caplog.at_level(logging.WARNING, logger="lanet_vi"):
            draw_community_boundaries(ax, result, positions)
        assert len(ax.collections) == 1
        assert len(ax.collections[0].get_paths()) == 1  # the collinear community only
        assert "Could not compute" not in caplog.text
    finally:
        plt.close(fig)


def test_community_colors_are_keyed_by_id_not_position():
    """Community ids need not be 0..n-1: colors follow the id, not the list index."""
    result = CommunityResult(
        algorithm="test",
        communities=[Community(id=7, nodes=[0, 1]), Community(id=3, nodes=[2])],
        node_to_community={0: 7, 1: 7, 2: 3},
    )
    positions = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0)}
    colors = assign_node_colors_by_community(result, positions)
    palette = get_community_colors(2)
    assert colors[0] == colors[1] == palette[0]
    assert colors[2] == palette[1]


def test_draw_community_circles_enclose_their_nodes():
    """One circle per community with placed nodes, each containing its nodes."""
    result, positions = _two_triangles_result()
    del positions[6]  # community 2 has no placed node and gets no circle
    fig, ax = plt.subplots()
    try:
        draw_community_circles(ax, result, positions, padding=0.5)
        circles = [patch for patch in ax.patches if isinstance(patch, plt.Circle)]
        assert len(circles) == 2
        for community, circle in zip(result.communities[:2], circles, strict=True):
            center = np.array(circle.center)
            for node in community.nodes:
                assert np.linalg.norm(np.array(positions[node]) - center) <= circle.radius + 1e-9
    finally:
        plt.close(fig)
