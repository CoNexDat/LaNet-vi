"""Tests for the k-connectivity of the k-core clusters (the C++ -kconn)."""

import random

import networkx as nx
import numpy as np
import pytest

from lanet_vi.decomposition.kconnectivity import (
    cluster_conditions,
    compute_kconnectivity,
    diameter_at_most_two,
)
from lanet_vi.visualization.lanet_layout import build_component_tree, clusters_by_index


def clusters_of(graph: nx.Graph, seed: int = 0) -> dict[int, list[list[int]]]:
    """Return the clusters the layout would draw, keyed by shell index."""
    core = nx.core_number(graph)
    root = build_component_tree(
        graph, core, lambda u, v: min(core[u], core[v]), np.random.default_rng(seed)
    )
    return clusters_by_index(root)


def kconn(graph: nx.Graph, kind: str = "wide", seed: int = 0) -> dict[int, int]:
    """Compute the k-connectivity of ``graph`` from its layout clusters."""
    return compute_kconnectivity(graph, clusters_of(graph, seed), kind)


def test_diameter_at_most_two():
    """A star and a clique have diameter <= 2; a path of four nodes does not."""
    assert diameter_at_most_two(nx.star_graph(5), range(6))
    assert diameter_at_most_two(nx.complete_graph(4), range(4))
    assert diameter_at_most_two(nx.path_graph(1), [0])
    assert not diameter_at_most_two(nx.path_graph(4), range(4))
    # Disconnected: not within two hops
    assert not diameter_at_most_two(nx.empty_graph(2), range(2))


def test_clique_is_k_connected_at_its_shell():
    """A clique is its own seed: every node gets the shell index, in both types."""
    graph = nx.complete_graph(5)
    for kind in ("wide", "strict"):
        assert kconn(graph, kind) == dict.fromkeys(range(5), 4)


def test_pendant_node_joins_at_one():
    """A leaf hanging from the seed touches C: it is 1-connected (|B| >= 1 at k = 1)."""
    graph = nx.complete_graph(5)
    graph.add_edge(0, 5)
    for kind in ("wide", "strict"):
        result = kconn(graph, kind)
        assert result[5] == 1
        assert all(result[v] == 4 for v in range(5))


def test_two_cliques_through_a_connector_one_is_left_out():
    """Two K5 joined by a degree-2 node (shell 2): the seed clique gets 4, the connector 2.

    The other clique is examined (and rejected: it touches nothing) before the
    connector joins C and there is no lower shell to try it again, so it stays 0 in
    both types, as the C++ walk leaves it.
    """
    graph = nx.disjoint_union(nx.complete_graph(5), nx.complete_graph(5))
    graph.add_edges_from([(0, 10), (5, 10)])
    assert nx.core_number(graph)[10] == 2
    for kind in ("wide", "strict"):
        result = kconn(graph, kind)
        assert result[10] == 2
        values = sorted(result[v] for v in range(10))
        assert values == [0] * 5 + [4] * 5


def test_wide_gives_a_lower_index_to_a_pending_cluster():
    """A shell-3 cluster left pending by the walk joins later at 2 in wide, not in strict.

    A K5 seed, a K4 attached to it by two nodes (the K4 stays shell 3, its diameter
    is 1 but the seed was found first) and a shell-2 ring: when k = 2 is processed the
    K4 touches C with two nodes, |B| = 2 >= 2, so it joins with k-connectivity 2.
    """
    graph = nx.complete_graph(5)
    k4 = [5, 6, 7, 8]
    graph.add_edges_from((u, v) for u in k4 for v in k4 if u < v)
    graph.add_edges_from([(5, 0), (6, 1)])
    graph.add_edges_from([(9, 10), (10, 11), (11, 9), (9, 2)])  # a triangle, shell 2
    core = nx.core_number(graph)
    assert core[5] == 3 and core[9] == 2 and core[0] == 4
    wide = kconn(graph, "wide")
    strict = kconn(graph, "strict")
    assert all(wide[v] == 4 for v in range(5))
    assert all(wide[v] == 2 for v in k4)
    assert all(strict[v] == 4 for v in range(5))
    assert all(strict[v] == 0 for v in k4)


def test_strict_finds_no_seed_when_the_top_core_has_diameter_three():
    """Karate club: strict never seeds, wide seeds through the minimum edge cut.

    The 4-core has diameter 3 and no other cluster reaches a minimum degree of its shell.
    """
    graph = nx.Graph(nx.karate_club_graph().edges())
    core = nx.core_number(graph)
    assert nx.diameter(graph.subgraph(v for v in graph if core[v] == 4)) == 3
    strict = kconn(graph, "strict")
    assert set(strict.values()) == {0}
    wide = kconn(graph, "wide")
    assert all(wide[v] == 4 for v in graph if core[v] == 4)
    assert all(wide[v] > 0 for v in graph)


def test_values_never_exceed_the_shell_and_strict_never_goes_below():
    """On random graphs: 0 <= kconn <= shell; strict is either 0 or the shell index."""
    rng = random.Random(3)
    for _ in range(30):
        n = rng.randint(8, 40)
        graph = nx.gnp_random_graph(n, rng.uniform(0.1, 0.4), seed=rng.randint(0, 10**6))
        graph.remove_nodes_from(list(nx.isolates(graph)))
        if graph.number_of_nodes() < 3:
            continue
        core = nx.core_number(graph)
        for kind in ("wide", "strict"):
            result = kconn(graph, kind, seed=rng.randint(0, 100))
            assert set(result) == set(graph)
            for v, value in result.items():
                assert 0 <= value <= core[v]
                if kind == "strict":
                    assert value in (0, core[v])


def test_conditions_first_frontier_then_phi():
    """The frontier test accepts a cluster touching C everywhere; phi rejects a path."""
    # C = {0}; cluster = path 1-2-3 where only 1 touches 0
    graph = nx.path_graph([0, 1, 2, 3])
    connected = {0: 5}
    # k = 2: contracted diameter of {1, 2, 3} + virtual(1) is 3 > 2 -> rejected
    assert not cluster_conditions(graph, [1, 2, 3], 2, connected)
    # k = 1: diameter skipped; |B| = 1 >= 1 -> accepted
    assert cluster_conditions(graph, [1, 2, 3], 1, connected)
    # A triangle touching C at one node, k = 2: diameter fine, |B| = 1 < 2, phi = 1
    graph = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 1)])
    assert not cluster_conditions(graph, [1, 2, 3], 2, connected)
    # Two touching nodes: |B| = 2 >= 2
    graph.add_edge(0, 2)
    assert cluster_conditions(graph, [1, 2, 3], 2, connected)


def test_self_loops_are_ignored_and_kind_is_checked():
    """A self-loop changes nothing; an unknown kind is an error."""
    graph = nx.complete_graph(4)
    clusters = clusters_of(graph)
    plain = compute_kconnectivity(graph, clusters)
    graph.add_edge(0, 0)
    assert compute_kconnectivity(graph, clusters) == plain
    with pytest.raises(ValueError, match="kind"):
        compute_kconnectivity(graph, clusters, "loose")


def test_empty_graph_and_empty_cluster_map():
    """No clusters at all: every node (if any) is 0."""
    assert compute_kconnectivity(nx.Graph(), {}) == {}
    graph = nx.empty_graph(3)
    assert compute_kconnectivity(graph, {}) == {0: 0, 1: 0, 2: 0}


def test_clusters_by_index_follows_the_tree_walk():
    """Parents' clusters come before their children's; indices without clusters are absent."""
    graph = nx.complete_graph(5)
    graph.add_edges_from([(0, 5), (5, 6)])  # a path: 5 and 6 have shell 1
    clusters = clusters_of(graph)
    assert sorted(clusters) == [1, 4]
    assert [sorted(c) for c in clusters[4]] == [list(range(5))]
    assert [sorted(c) for c in clusters[1]] == [[5, 6]]
    # The walk order: the root's clusters (index 1) come first
    core = nx.core_number(graph)
    root = build_component_tree(
        graph, core, lambda u, v: min(core[u], core[v]), np.random.default_rng(0)
    )
    assert list(clusters_by_index(root)) == [1, 4]
