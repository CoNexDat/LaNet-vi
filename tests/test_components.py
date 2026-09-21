"""Tests for the one-pass components per level (decomposition/components.py, union_find.py)."""

import networkx as nx
import numpy as np

from lanet_vi.decomposition.components import components_by_index
from lanet_vi.decomposition.kcores import _core_number
from lanet_vi.union_find import UnionFind


def _per_level_components(graph: nx.Graph, index: dict) -> dict:
    """Build the pieces by their definition: connected components of every level's subgraph."""
    pieces = nx.weakly_connected_components if graph.is_directed() else nx.connected_components
    return {
        k: frozenset(
            frozenset(c) for c in pieces(graph.subgraph(v for v in index if index[v] == k))
        )
        for k in set(index.values())
    }


def test_union_find_merges_and_tracks_sizes():
    """Sets merge transitively and the representative's size is the set's."""
    sets = UnionFind(5)
    sets.union(0, 1)
    sets.union(3, 4)
    sets.union(1, 4)
    assert len({sets.find(i) for i in (0, 1, 3, 4)}) == 1
    assert sets.find(2) == 2
    assert sets.size[sets.find(0)] == 4


def test_components_ordered_by_index_then_first_node_with_running_ids():
    """Highest index first, pieces by their first node in graph order, nodes in that order."""
    G = nx.Graph()
    G.add_nodes_from([5, 1, 2, 3, 4, 6])
    G.add_edges_from([(1, 2), (2, 3), (3, 1), (4, 5), (4, 6)])
    index = {1: 2, 2: 2, 3: 2, 4: 1, 5: 1, 6: 1, 7: 9}  # 7 is not in the graph
    comps = components_by_index(G, index)
    assert [(c.component_id, c.shell_index, c.nodes, c.size) for c in comps] == [
        (0, 2, [1, 2, 3], 3),
        (1, 1, [5, 4, 6], 3),
    ]
    assert all(c.dense_index is None for c in comps)
    dense = components_by_index(G, index, dense=True)
    assert [(c.dense_index, c.shell_index) for c in dense] == [(2, None), (1, None)]


def test_components_split_a_level_that_only_touches_through_another_level():
    """Two index-1 nodes joined only through an index-2 node are two pieces."""
    G = nx.Graph([(0, 1), (1, 2), (2, 0), (0, 3), (1, 4)])
    comps = components_by_index(G, nx.core_number(G))
    assert [(c.shell_index, sorted(c.nodes)) for c in comps] == [
        (2, [0, 1, 2]),
        (1, [3]),
        (1, [4]),
    ]


def test_directed_graphs_use_weak_connectivity():
    """Arcs in either direction connect the pieces, as weakly_connected_components did."""
    G = nx.DiGraph([(0, 1), (2, 1)])  # no directed path 0 -> 2, one weak piece
    comps = components_by_index(G, {0: 1, 1: 1, 2: 1})
    assert [sorted(c.nodes) for c in comps] == [[0, 1, 2]]


def test_components_match_the_per_level_construction_on_random_graphs():
    """Simple, directed and multigraph inputs give the pieces of the per-level subgraphs."""
    rng = np.random.default_rng(11)
    for trial in range(60):
        n = int(rng.integers(0, 40))
        directed = trial % 3 == 1
        G = nx.gnp_random_graph(n, 0.1, seed=trial, directed=directed)
        if trial % 3 == 2 and G.number_of_edges():
            G = nx.MultiGraph(G)
            G.add_edges_from(list(G.edges())[:3])
        index = _core_number(G)
        comps = components_by_index(G, index)
        got: dict[int, set] = {}
        for c in comps:
            got.setdefault(c.shell_index, set()).add(frozenset(c.nodes))
        assert {k: frozenset(v) for k, v in got.items()} == _per_level_components(G, index)
        assert [c.component_id for c in comps] == list(range(len(comps)))


def test_core_number_matches_networkx_on_simple_and_directed_graphs():
    """The bucket peeling gives nx.core_number's numbers, mutual arcs counted twice."""
    for seed in range(30):
        for directed in (False, True):
            G = nx.gnp_random_graph(int(30 + seed), 0.12, seed=seed, directed=directed)
            assert _core_number(G) == dict(nx.core_number(G))
    assert _core_number(nx.Graph()) == {}
    assert _core_number(nx.empty_graph(3)) == {0: 0, 1: 0, 2: 0}
