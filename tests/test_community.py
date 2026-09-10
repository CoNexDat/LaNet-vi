"""Tests for community detection data models and colouring."""

import networkx as nx

from lanet_vi.community.base import Community, CommunityResult
from lanet_vi.community.louvain import detect_communities_louvain
from lanet_vi.visualization.community_viz import get_community_colors


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
    """Community colours are RGB triples in [0, 1], including beyond 20 communities."""
    for n in (5, 25, 60):
        colors = get_community_colors(n)
        assert len(colors) == n
        for color in colors:
            assert len(color) == 3
            assert all(0.0 <= c <= 1.0 for c in color)
