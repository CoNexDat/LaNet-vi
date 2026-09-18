"""Tests for the LaNet-vi placement port (visualization/lanet_layout.py)."""

import math

import networkx as nx
import numpy as np
import pytest

from lanet_vi.visualization.lanet_layout import (
    LayoutParameters,
    build_component_tree,
    circular_average,
    compute_lanet_layout,
    node_radius,
    place_in_circular_sector,
)


def _min_core(core: dict[int, int]):  # noqa: ANN202
    return lambda u, v: min(core[u], core[v])


def _two_k4_bridged() -> nx.Graph:
    """Two K4 (3-cores) joined through node 8 (2-core), a pendant 9 (1-core), isolated 10."""
    G = nx.complete_graph(4)
    G.add_edges_from(nx.complete_graph(range(4, 8)).edges())
    G.add_edges_from([(3, 8), (8, 4), (0, 9)])
    G.add_node(10)
    return G


def test_component_tree_nests_cores_and_clusters():
    """Root (0) > 1-core > 2-core > two 3-core leaves; clusters hold each shell's nodes."""
    G = _two_k4_bridged()
    core = nx.core_number(G)
    root = build_component_tree(G, core, _min_core(core), np.random.default_rng(0))

    assert root.index == 0 and root.size == 11
    assert [sorted(c) for c in root.clusters] == [[10]]  # the isolated node is the 0-shell
    (one,) = root.children
    assert one.index == 1 and one.size == 10 and one.shell_cardinal == 1
    assert [sorted(c) for c in one.clusters] == [[9]]
    (two,) = one.children
    assert two.index == 2 and two.size == 9 and [sorted(c) for c in two.clusters] == [[8]]
    assert sorted(child.size for child in two.children) == [4, 4]
    assert all(child.index == 3 and not child.children for child in two.children)
    assert sorted(sorted(c) for child in two.children for c in child.clusters) == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]


def test_clusters_are_connected_within_the_shell():
    """Two 1-shell nodes that only touch through a higher node form two clusters."""
    # 0-1-2 triangle (core 2); 3 and 4 hang from different triangle nodes (core 1)
    G = nx.Graph([(0, 1), (1, 2), (2, 0), (0, 3), (1, 4)])
    core = nx.core_number(G)
    root = build_component_tree(G, core, _min_core(core), np.random.default_rng(0))
    (one,) = root.children
    assert sorted(sorted(c) for c in one.clusters) == [[3], [4]]
    assert one.shell_cardinal == 2


def test_every_node_is_placed_and_shells_are_concentric_rings():
    """Karate club: rings one unit apart, ordered by core, the top core innermost."""
    G = nx.Graph(nx.karate_club_graph().edges())
    core = nx.core_number(G)
    params = LayoutParameters(epsilon=0.18, gamma=1.0, u=1.0)

    layout = compute_lanet_layout(G, core, params, seed=3)

    assert set(layout.positions) == set(G.nodes())
    root = layout.root
    chain = [root]
    while chain[-1].children:
        assert len(chain[-1].children) == 1  # a single nested chain: all concentric
        chain.append(chain[-1].children[0])
    assert [c.index for c in chain] == [0, 1, 2, 3, 4]
    assert all((c.x, c.y) == (0.0, 0.0) for c in chain)
    leaf = chain[-1]
    for depth, comp in enumerate(reversed(chain)):
        assert comp.ratio == pytest.approx(leaf.central_core_ratio + depth)

    by_core: dict[int, list[float]] = {}
    for node, (x, y) in layout.positions.items():
        by_core.setdefault(core[node], []).append(math.hypot(x, y))
    for k in (1, 2, 3):
        ratio = leaf.central_core_ratio + (4 - k)
        assert min(by_core[k]) >= ratio * (1 - params.epsilon) - 1e-9
        assert max(by_core[k]) <= ratio + 1e-9
    assert max(by_core[4]) <= leaf.central_core_ratio + 1e-9  # cliques inside the core disc
    assert layout.frame == pytest.approx(params.gamma * params.u * chain[1].ratio)


def test_sibling_components_get_distinct_centres_and_scales():
    """Two 3-cores inside one 2-core are offset from the parent centre (formulas 3-5)."""
    G = _two_k4_bridged()
    core = nx.core_number(G)
    layout = compute_lanet_layout(G, core, LayoutParameters(delta=1.3), seed=0)
    (one,) = layout.root.children
    (two,) = one.children
    a, b = two.children
    assert (a.x, a.y) != (b.x, b.y)
    for child in (a, b):
        assert child.rho == pytest.approx(1 - child.size / 8)
        assert child.u == pytest.approx(math.sqrt(child.size / 8) * two.u / 1.3)
        assert math.hypot(child.x - two.x, child.y - two.y) == pytest.approx(
            two.ratio * two.u * child.rho
        )
    assert one.u == layout.root.u and two.u == one.u  # only children inherit the scale


def test_layout_is_deterministic_per_seed():
    """Same seed, same picture; another seed, another picture."""
    G = nx.Graph(nx.karate_club_graph().edges())
    core = nx.core_number(G)
    first = compute_lanet_layout(G, core, LayoutParameters(), seed=7).positions
    again = compute_lanet_layout(G, core, LayoutParameters(), seed=7).positions
    other = compute_lanet_layout(G, core, LayoutParameters(), seed=8).positions
    assert first == again
    assert first != other


def test_top_core_cliques_share_the_disc():
    """A top core that is a single clique goes on the arc at 0.9 of the core radius."""
    G = nx.complete_graph(5)
    core = nx.core_number(G)
    layout = compute_lanet_layout(G, core, LayoutParameters(gamma=1.0), seed=0)
    leaf = layout.root
    while leaf.children:
        leaf = leaf.children[0]
    # On the arc at 0.9 R, shifted by 0.1 R towards the sector's middle: within R
    radii = [math.hypot(x, y) for x, y in layout.positions.values()]
    assert all(0.8 * leaf.ratio - 1e-9 <= r <= leaf.ratio + 1e-9 for r in radii)


def test_no_cliques_places_top_core_at_random_inside_the_disc():
    """--no-cliques: the top core is spread uniformly inside its disc."""
    G = nx.complete_graph(6)
    core = nx.core_number(G)
    layout = compute_lanet_layout(G, core, LayoutParameters(gamma=1.0, no_cliques=True), seed=1)
    leaf = layout.root
    while leaf.children:
        leaf = leaf.children[0]
    radii = [math.hypot(x, y) for x, y in layout.positions.values()]
    assert max(radii) <= leaf.ratio + 1e-9
    assert len(set(round(r, 6) for r in radii)) > 1


def test_weighted_graph_uses_edge_weights_without_crashing():
    """Weighted mode reads the weight attribute for rho and phi."""
    G = nx.Graph(nx.karate_club_graph().edges())
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0 + (u + v) % 3
    core = nx.core_number(G)
    layout = compute_lanet_layout(G, core, LayoutParameters(weighted=True), seed=0)
    assert set(layout.positions) == set(G.nodes())


def test_edge_index_drives_the_components():
    """With an edge index that isolates an edge, its endpoints split into two components."""
    G = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)])  # two triangles sharing 2
    # As in k-dense: a node's index is the maximum of its edges' indices
    index = {0: 2, 1: 2, 2: 2, 3: 1, 4: 1}
    weak = {(2, 3), (3, 4), (2, 4)}

    def edge_index(u: int, v: int) -> int:
        return 1 if ((u, v) if u < v else (v, u)) in weak else 2

    root = build_component_tree(G, index, edge_index, np.random.default_rng(0))
    (one,) = root.children
    assert len(one.children) == 1  # only the strong triangle is connected above index 1
    (two,) = one.children
    assert sorted(v for c in two.clusters for v in c) == [0, 1, 2]
    assert sorted(v for c in one.clusters for v in c) == [3, 4]


def test_empty_graph():
    """No nodes: nothing placed, a sane frame."""
    layout = compute_lanet_layout(nx.Graph(), {}, LayoutParameters(), seed=0)
    assert layout.positions == {} and layout.frame > 0


def test_place_in_circular_sector_walks_the_u_path():
    """Out along the radius, along the arc, back in."""
    # Path length 2 + 1 (out, arc of aperture 1 at radius 1, in), sampled at quarters
    assert place_in_circular_sector(1.0, 1.0, 0, 4, 0.0, unique=False) == (0.0, 0.0)
    rho, phi = place_in_circular_sector(1.0, 1.0, 1, 4, 0.0, unique=False)
    assert (rho, phi) == (pytest.approx(0.75), 0.0)
    rho, phi = place_in_circular_sector(1.0, 1.0, 2, 4, 0.0, unique=False)
    assert (rho, phi) == (pytest.approx(1.0), pytest.approx(0.5))
    rho, phi = place_in_circular_sector(1.0, 1.0, 3, 4, 0.0, unique=False)
    assert (rho, phi) == (pytest.approx(0.75), pytest.approx(1.0))
    rho, phi = place_in_circular_sector(1.0, 1.0, 0, 2, 0.0, unique=True)
    assert (rho, phi) == (1.0, 0.0)


def test_circular_average_takes_the_short_arc():
    """Angles either side of 0 average near 0, not near pi."""
    result = circular_average(0.1, 1.0, 2 * math.pi - 0.1, 1.0)
    assert result % (2 * math.pi) == pytest.approx(0.0, abs=1e-6) or result == pytest.approx(
        2 * math.pi, abs=1e-6
    )
    assert circular_average(0.0, 1.0, 1.0, 3.0) == pytest.approx(0.75)
    assert circular_average(1.0, 3.0, 0.0, 1.0) == pytest.approx(0.75)


def test_node_radius_follows_the_cpp_formula():
    """0.4 for the largest degree, (log(1+d)/log(dmax))^0.7 below, strength when weighted."""
    assert node_radius(100, 100) == pytest.approx(0.4 * (math.log(101) / math.log(100)) ** 0.7)
    assert node_radius(1, 100) < node_radius(50, 100) < node_radius(100, 100)
    assert node_radius(1, 1) == 0.4
    assert node_radius(0, 10, 5.0, 100.0, weighted=True) == pytest.approx(
        0.4 * math.log(6.0) / math.log(100.0)
    )


def test_higher_neighbour_in_another_branch_is_tolerated():
    """k-dense-like indices: a node's high neighbour may be reachable only through a weak edge."""
    # Triangle 0-1-2 (index 3 nodes, index-3 edges) and node 3 with index 3 attached to 0 by
    # an index-2 edge; node 4 (index 2) attached to 3 by an index-2 edge.
    G = nx.Graph([(0, 1), (1, 2), (2, 0), (0, 3), (3, 4)])
    index = {0: 3, 1: 3, 2: 3, 3: 3, 4: 2}
    strong = {(0, 1), (1, 2), (0, 2)}

    def edge_index(u: int, v: int) -> int:
        return 3 if ((u, v) if u < v else (v, u)) in strong else 2

    layout = compute_lanet_layout(G, index, LayoutParameters(), seed=0, edge_index=edge_index)
    assert set(layout.positions) == set(G.nodes())


def test_single_isolated_node_and_edgeless_graph():
    """No edges: the log(max strength) guard keeps the layout finite."""
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2])
    layout = compute_lanet_layout(G, {0: 0, 1: 0, 2: 0}, LayoutParameters(), seed=0)
    assert set(layout.positions) == {0, 1, 2}
    assert all(math.isfinite(x) and math.isfinite(y) for x, y in layout.positions.values())
    single = compute_lanet_layout(nx.Graph([(0, 0)]), {0: 0}, LayoutParameters(), seed=0)
    assert set(single.positions) == {0}


def test_unweighted_average_matches_formula_one():
    """Rho = R (1 - eps) + eps R avg with avg = sum(max - k_nb + 1) / (L (max - k))."""
    # Node 5 (core 2) hangs from 0 and 1 of a K5 (core 4): L = 2, max = 4, depth = 2,
    # sum = 2 * (4 - 4 + 1) = 2 -> avg = 0.5 (an extra / L that once slipped in gave 0.25)
    G = nx.complete_graph(5)
    G.add_edges_from([(5, 0), (5, 1)])
    core = nx.core_number(G)
    assert core[5] == 2
    params = LayoutParameters(epsilon=0.5, gamma=1.0, u=1.0, no_cliques=True)
    layout = compute_lanet_layout(G, core, params, seed=0)
    comp = layout.root
    while comp.index != 2:
        comp = comp.children[0]
    x, y = layout.positions[5]
    assert math.hypot(x - comp.x, y - comp.y) == pytest.approx(comp.ratio * (0.5 + 0.5 * 0.5))


def test_deep_core_hierarchy_does_not_hit_the_recursion_limit():
    """A chain of 1500 nested cores is placed iteratively."""
    # Nested cliques would be huge; fake the indices on a path instead: node i has index i
    # and the edge (i, i+1) index i, so every node is its own nested component
    n = 1500
    G = nx.path_graph(n)
    index = {i: i for i in range(n)}
    layout = compute_lanet_layout(
        G, index, LayoutParameters(), seed=0, edge_index=lambda u, v: min(index[u], index[v])
    )
    assert len(layout.positions) == n
    depth = 0
    comp = layout.root
    while comp.children:
        comp = comp.children[0]
        depth += 1
    assert depth == n - 1


def test_greedy_cliques_scale_to_a_large_top_core():
    """A 1000-node top core (two K500 joined by an edge) is partitioned quickly."""
    from lanet_vi.visualization.lanet_layout import _greedy_cliques

    G = nx.complete_graph(500)
    G.add_edges_from(nx.complete_graph(range(500, 1000)).edges())
    G.add_edge(0, 500)
    cliques = _greedy_cliques(list(G.nodes()), G)
    assert sorted(len(c) for c in cliques) == [500, 500]
    assert sorted(v for c in cliques for v in c) == list(range(1000))


def test_multigraph_weights_are_summed_for_the_weighted_geometry():
    """Parallel edges of a MultiGraph count with their summed weight, not as weight 1."""
    from lanet_vi.visualization.lanet_layout import _merge_parallel_edges, _Placer

    multi = nx.MultiGraph()
    multi.add_weighted_edges_from([(0, 1, 100.0), (0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)])
    index = {0: 2, 1: 2, 2: 2}
    layout = compute_lanet_layout(multi, index, LayoutParameters(weighted=True), seed=0)
    assert set(layout.positions) == {0, 1, 2}

    # The placer sees a simple graph whose strengths reflect the 100-weight edge
    placer = _Placer(
        _merge_parallel_edges(multi),
        index,
        LayoutParameters(weighted=True),
        np.random.default_rng(0),
        dict(multi.degree()),
    )
    assert placer.log_max_strength == pytest.approx(math.log(102.0))
    assert placer.degree[0] == 3  # multiplicity kept for the degree-based radii
