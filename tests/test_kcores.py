"""Tests for k-core decomposition."""

from pathlib import Path

import networkx as nx
import pytest

from lanet_vi.decomposition.kcores import compute_kcores
from lanet_vi.io.readers import read_edge_list
from lanet_vi.models.config import DecompositionConfig, StrengthIntervalMethod


def test_kcores_simple_graph():
    """Test k-core decomposition on a simple graph."""
    # Create a simple graph with known k-core structure
    G = nx.Graph()
    # Triangle (3-core)
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    # Attached nodes (lower cores)
    G.add_edge(0, 3)
    G.add_edge(3, 4)

    result = compute_kcores(G)

    assert result.decomp_type == "kcores"
    assert result.max_index >= 2  # Triangle has at least 2-core
    assert result.min_index >= 0
    assert len(result.node_indices) == 5  # 5 nodes total


def test_kcores_karate_club():
    """Test k-core on Zachary's karate club graph."""
    G = nx.karate_club_graph()

    result = compute_kcores(G)

    assert result.decomp_type == "kcores"
    assert result.max_index > 0
    assert len(result.node_indices) == 34  # Karate club has 34 nodes


def test_kcores_empty_graph():
    """Test k-core on empty graph."""
    G = nx.Graph()

    result = compute_kcores(G)

    assert result.decomp_type == "kcores"
    assert result.max_index == 0
    assert result.min_index == 0
    assert len(result.node_indices) == 0


def test_kcores_weighted_graph():
    """Test k-core on weighted graph."""
    G = nx.Graph()
    G.add_edge(0, 1, weight=2.0)
    G.add_edge(1, 2, weight=3.0)
    G.add_edge(2, 0, weight=1.5)

    config = DecompositionConfig(granularity=3)
    result = compute_kcores(G, config)

    assert result.decomp_type == "kcores"
    assert result.p_function is not None  # Weighted graph should have p-function
    assert len(result.p_function) > 0


# --- multigraph, self-loop and weighted-detection behaviour (#22)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "edges.txt"
    path.write_text(text)
    return path


def test_multigraph_kcores_count_parallel_edges(tmp_path: Path):
    """Two parallel edges between 1 and 2 give them degree 2, so they form a 2-core."""
    path = _write(tmp_path, "1 2\n1 2\n2 3\n")
    G = read_edge_list(path, multigraph=True)

    result = compute_kcores(G)

    assert result.node_indices == {1: 2, 2: 2, 3: 1}


def test_multigraph_kcores_match_simple_graph_when_no_parallel_edges():
    """A MultiGraph without parallel edges gives the same core numbers as nx.Graph."""
    simple = nx.karate_club_graph()
    multi = nx.MultiGraph(simple)

    assert compute_kcores(multi).node_indices == compute_kcores(simple).node_indices


def test_kcores_ignore_self_loops():
    """Self-loops do not change core numbers and do not crash."""
    G = nx.karate_club_graph()
    expected = compute_kcores(G).node_indices
    G.add_edge(0, 0)

    assert compute_kcores(G).node_indices == expected


def test_kcores_weighted_graph_with_only_self_loops():
    """A weighted graph whose only edges are self-loops does not crash."""
    G = nx.Graph()
    G.add_edge(1, 1, weight=2.0)
    G.add_node(2)

    result = compute_kcores(G)

    assert result.node_indices == {1: 0, 2: 0}


def test_multidigraph_kcores_count_both_directions():
    """For a directed multigraph, removing a node decrements predecessors too."""
    x, y, u, v = 0, 1, 2, 3
    G = nx.MultiDiGraph()
    G.add_edges_from([(x, y), (y, x), (x, u), (u, v)])

    result = compute_kcores(G)  # type: ignore[arg-type]

    assert result.node_indices == {x: 2, y: 2, u: 1, v: 1}


def test_weighted_multigraph_sums_parallel_weights():
    """Parallel weighted edges are merged by summing before the strength-based cores."""
    from lanet_vi.decomposition.kcores import _as_weighted_simple_graph

    G = nx.MultiGraph()
    G.add_weighted_edges_from([(1, 2, 0.5), (1, 2, 1.5), (2, 3, 1.0)])

    merged = _as_weighted_simple_graph(G)
    assert not merged.is_multigraph()
    assert merged[1][2]["weight"] == 2.0

    result = compute_kcores(G)  # runs the weighted path through the merge
    assert result.p_function is not None
    assert set(result.node_indices) == {1, 2, 3}


def test_weighted_flag_wins_over_late_weight_attribute():
    """weighted=True forces the strength path even if the first edges carry no weight."""
    G = nx.path_graph(150)
    G.add_edge(0, 149, weight=3.0)

    assert compute_kcores(G, weighted=False).p_function is None
    assert compute_kcores(G).p_function is not None  # autodetect scans every edge
    assert compute_kcores(G, weighted=True).p_function is not None


def test_weighted_digraph_strength_counts_both_directions():
    """Reciprocal weighted arcs are summed into one undirected weight."""
    from lanet_vi.decomposition.kcores import _as_weighted_simple_graph

    G = nx.DiGraph()
    G.add_weighted_edges_from([(1, 2, 1.0), (2, 1, 2.0)])

    merged = _as_weighted_simple_graph(G)
    assert not merged.is_directed()
    assert merged[1][2]["weight"] == 3.0


def test_kcores_weighted_flag_on_edgeless_graph():
    """weighted=True on a graph without edges returns zero cores instead of crashing."""
    G = nx.Graph()
    G.add_nodes_from([1, 2])

    assert compute_kcores(G, weighted=True).node_indices == {1: 0, 2: 0}


def test_network_directed_multigraph_kcores_end_to_end(tmp_path: Path):
    """--directed --multigraph works through Network.decompose (weak components)."""
    from lanet_vi.core.network import Network
    from lanet_vi.models.config import DecompositionType, GraphConfig, LaNetConfig

    path = _write(tmp_path, "1 2\n1 2\n2 3\n3 1\n")
    net = Network.from_edge_list(
        path, LaNetConfig(graph=GraphConfig(directed=True, multigraph=True))
    )

    result = net.decompose(DecompositionType.KCORES)

    assert result.components
    assert sum(c.size for c in result.components) == 3


def test_network_autodetects_weights_without_config():
    """Network(G) with weighted edges still takes the strength path by default."""
    from lanet_vi.core.network import Network

    G = nx.karate_club_graph()
    nx.set_edge_attributes(G, 1.0, "weight")

    assert Network(G).decompose().p_function is not None


# --- weighted (strength-based) k-cores: the C++ peeling (#20)


def _p_index(p_function: list[float], strength: float) -> int:
    """Smallest i with p_function[i] >= strength (last index if above all boundaries)."""
    for i, boundary in enumerate(p_function):
        if boundary >= strength:
            return i
    return len(p_function) - 1


def _brute_force_weighted_cores(G: nx.Graph, p_function: list[float]) -> dict[int, int]:
    """Generalised core by fixed point, independent of the peeling order.

    A node has index >= k iff it belongs to the maximal subgraph in which every node
    receives, from the other nodes of the subgraph, a strength whose interval is >= k.
    """
    index = {v: 0 for v in G}
    for k in range(1, len(p_function)):
        survivors = set(G)
        changed = True
        while changed:
            changed = False
            for v in list(survivors):
                strength = sum(G[v][x].get("weight", 1.0) for x in G[v] if x in survivors)
                if _p_index(p_function, strength) < k:
                    survivors.remove(v)
                    changed = True
        for v in survivors:
            index[v] = k
    return index


def _random_weighted_graph(seed: int) -> nx.Graph:
    import random

    rng = random.Random(seed)
    G = nx.gnp_random_graph(rng.randint(6, 30), rng.uniform(0.15, 0.5), seed=seed)
    for u, v in G.edges():
        G[u][v]["weight"] = rng.choice([0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
    return G


def test_weighted_cores_peel_instead_of_binning_total_strength():
    """Issue #20: a hub whose strength comes from weak leaves must drop to their shell."""
    # Triangle of weight-3 edges plus hub 3 tied to it by a weight-1 edge and carrying
    # five leaves of weight 1: the hub's total strength (6) equals a triangle node's.
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 3.0), (1, 2, 3.0), (2, 0, 3.0), (0, 3, 1.0)])
    G.add_weighted_edges_from([(3, leaf, 1.0) for leaf in range(4, 9)])
    config = DecompositionConfig(granularity=3, maximum_strength=6.0)  # 0, 2, 4, 6

    result = compute_kcores(G, config, weighted=True)

    # Leaves: strength 1 -> interval 1. Once they are peeled the hub keeps only the
    # weight-1 edge to node 0 -> interval 1 (binning alone would have given it 3).
    assert result.p_function == [0.0, 2.0, 4.0, 6.0]
    assert result.node_indices == {0: 3, 1: 3, 2: 3, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}


def test_weighted_cores_indices_start_at_one_and_isolated_nodes_at_zero():
    """Indices run 1..granularity (the C++ 3.0.1 ran 2..granularity+1 in two modes)."""
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 1.0)])
    G.add_node(9)

    result = compute_kcores(G, DecompositionConfig(granularity=4), weighted=True)

    assert result.max_index == 2  # node 1 has strength 2 = boundary 4 of 4 -> then peels
    assert result.node_indices[9] == 0
    assert min(v for n, v in result.node_indices.items() if n != 9) >= 1


@pytest.mark.parametrize("method", list(StrengthIntervalMethod)[:3])
@pytest.mark.parametrize("granularity", [-1, 4])
@pytest.mark.parametrize("seed", range(8))
def test_weighted_cores_match_fixed_point_oracle(
    method: StrengthIntervalMethod, granularity: int, seed: int
):
    """Every interval method and granularity agrees with the fixed-point definition."""
    G = _random_weighted_graph(seed)
    config = DecompositionConfig(strength_intervals=method, granularity=granularity)

    result = compute_kcores(G, config, weighted=True)

    assert result.p_function is not None
    assert result.node_indices == _brute_force_weighted_cores(G, result.p_function)


def test_weighted_default_granularity_is_the_maximum_degree():
    """No cap at 100: a hub of degree 120 gives 120 intervals, as in the C++."""
    G = nx.star_graph(120)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0

    result = compute_kcores(G, DecompositionConfig(), weighted=True)

    assert result.p_function is not None
    assert len(result.p_function) == 121  # 0.0 plus 120 boundaries


def test_weighted_maximum_strength_fixes_the_top_boundary():
    """maximum_strength normalises the intervals so different networks are comparable."""
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 2.0), (1, 2, 2.0)])

    result = compute_kcores(
        G, DecompositionConfig(granularity=4, maximum_strength=8.0), weighted=True
    )

    assert result.p_function == [0.0, 2.0, 4.0, 6.0, 8.0]


def test_weighted_custom_intervals_from_file(tmp_path: Path):
    """strength_intervals = custom reads the boundaries from a file (C++ 4.0.0)."""
    intervals = tmp_path / "intervals.txt"
    intervals.write_text("# boundaries\n1.5\n\n4\n")
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0), (2, 3, 5.0)])
    config = DecompositionConfig(
        strength_intervals=StrengthIntervalMethod.CUSTOM, strength_intervals_file=intervals
    )

    result = compute_kcores(G, config, weighted=True)

    assert result.p_function == [0.0, 1.5, 4.0]
    # Strengths: 0 -> 2, 1 -> 2, 2 -> 7 (above the last boundary: last index), 3 -> 5
    assert result.node_indices == _brute_force_weighted_cores(G, [0.0, 1.5, 4.0])
    assert result.node_indices[2] == 2 and result.node_indices[3] == 2


def test_weighted_custom_intervals_need_a_file(tmp_path: Path):
    """Custom intervals without a file, an empty file or unsorted boundaries are rejected."""
    G = nx.Graph()
    G.add_edge(0, 1, weight=1.0)

    with pytest.raises(ValueError, match="strength_intervals_file"):
        DecompositionConfig(strength_intervals=StrengthIntervalMethod.CUSTOM)

    empty = tmp_path / "empty.txt"
    empty.write_text("\n")
    unsorted = tmp_path / "unsorted.txt"
    unsorted.write_text("4\n1\n")
    for path, message in ((empty, "No strength intervals"), (unsorted, "non-decreasing")):
        config = DecompositionConfig(
            strength_intervals=StrengthIntervalMethod.CUSTOM, strength_intervals_file=path
        )
        with pytest.raises(ValueError, match=message):
            compute_kcores(G, config, weighted=True)


def test_weighted_two_column_file_uses_unit_weights(tmp_path: Path):
    """--weighted on a two-column edge list behaves like weight 1.0 everywhere (no NaN)."""
    path = tmp_path / "edges.txt"
    path.write_text("0 1\n1 2\n2 0\n2 3\n")
    G = read_edge_list(path, weighted=True)

    result = compute_kcores(G, DecompositionConfig(granularity=3), weighted=True)

    assert result.p_function == [0.0, 1.0, 2.0, 3.0]
    assert result.node_indices == {0: 2, 1: 2, 2: 2, 3: 1}


def test_weighted_custom_intervals_reject_non_finite_or_negative(tmp_path: Path):
    """nan, inf and negative boundaries would break the p-function contract."""
    G = nx.Graph()
    G.add_edge(0, 1, weight=1.0)
    for content in ("nan\n", "1\ninf\n", "-1\n2\n"):
        path = tmp_path / "bad.txt"
        path.write_text(content)
        config = DecompositionConfig(
            strength_intervals=StrengthIntervalMethod.CUSTOM, strength_intervals_file=path
        )
        with pytest.raises(ValueError, match="finite and non-negative"):
            compute_kcores(G, config, weighted=True)


def test_weighted_log_intervals_stay_monotone_with_small_maximum_strength():
    """maximum_strength below the smallest strength must not produce a descending scale."""
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 1.0)])
    config = DecompositionConfig(
        strength_intervals=StrengthIntervalMethod.EQUAL_LOG_SIZE,
        granularity=3,
        maximum_strength=0.5,
    )

    result = compute_kcores(G, config, weighted=True)

    assert result.p_function is not None
    assert result.p_function == sorted(result.p_function)
    assert result.p_function[-1] == 0.5


def test_weighted_all_zero_weights_do_not_crash_any_interval_method():
    """Zero weights are valid input; every method yields a flat scale and index 0."""
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 0.0), (1, 2, 0.0)])
    for method in list(StrengthIntervalMethod)[:3]:
        config = DecompositionConfig(strength_intervals=method, granularity=3)

        result = compute_kcores(G, config, weighted=True)

        assert result.p_function == [0.0, 0.0, 0.0, 0.0]
        assert set(result.node_indices.values()) == {0}


def test_weighted_negative_weights_are_rejected():
    """Negative weights would make the strength scale descend; refuse them clearly."""
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 1.0), (1, 2, -1.0)])
    with pytest.raises(ValueError, match="negative weight"):
        compute_kcores(G, DecompositionConfig(), weighted=True)


def test_weighted_hub_with_many_leaves_is_fast():
    """The residual strength is kept incrementally, so a hub is not rescanned per leaf."""
    import time

    G = nx.star_graph(20000)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0

    start = time.perf_counter()
    result = compute_kcores(G, DecompositionConfig(), weighted=True)  # granularity 20000
    elapsed = time.perf_counter() - start

    assert result.node_indices[0] == 1  # the hub ends with its leaves
    assert elapsed < 5.0
