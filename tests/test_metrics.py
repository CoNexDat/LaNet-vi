"""Tests for the partition metrics in ``lanet_vi.metrics``.

The information-theoretic quantities are checked against hand-computed values on tiny
partitions and against scikit-learn's implementations (which use natural logarithms, so
the mutual information is converted to bits) on random partitions.
"""

import math
import random

import networkx as nx
import pytest
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

from lanet_vi.decomposition.kcores import compute_kcores
from lanet_vi.metrics import (
    compare_decompositions,
    compare_partitions,
    compute_adjusted_rand_index,
    compute_mutual_information,
    compute_normalized_mutual_information,
    compute_partition_entropy,
)
from lanet_vi.metrics.information import compute_variation_of_information
from lanet_vi.metrics.similarity import compute_jaccard_similarity, compute_overlap_coefficient

BALANCED = {0: 0, 1: 0, 2: 1, 3: 1}
TRIVIAL = {0: 0, 1: 0, 2: 0, 3: 0}
SINGLETONS = {0: 0, 1: 1, 2: 2, 3: 3}


def _random_partitions(seed: int, n: int = 40) -> tuple[dict[int, int], dict[int, int]]:
    """Two random partitions of the same ``n`` nodes into a few clusters."""
    rng = random.Random(seed)
    return (
        {node: rng.randrange(3) for node in range(n)},
        {node: rng.randrange(4) for node in range(n)},
    )


def _labels(partition: dict[int, int]) -> list[int]:
    return [partition[node] for node in sorted(partition)]


# --- entropy ---------------------------------------------------------------------------


def test_entropy_of_balanced_partition_is_one_bit():
    """Two clusters of equal size carry exactly one bit."""
    assert compute_partition_entropy(BALANCED) == pytest.approx(1.0)


def test_entropy_is_zero_for_trivial_and_empty_partitions():
    """A single cluster (or no nodes at all) has no uncertainty."""
    assert compute_partition_entropy(TRIVIAL) == 0.0
    assert compute_partition_entropy({}) == 0.0


def test_entropy_is_maximal_for_singletons():
    """The n singletons give log2(n) bits, the maximum for n nodes."""
    assert compute_partition_entropy(SINGLETONS) == pytest.approx(2.0)


def test_entropy_of_unbalanced_partition():
    """Hand-computed value for sizes (3, 1)."""
    partition = {0: 0, 1: 0, 2: 0, 3: 1}
    expected = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
    assert compute_partition_entropy(partition) == pytest.approx(expected)


# --- mutual information ----------------------------------------------------------------


def test_mutual_information_of_identical_partitions_equals_entropy():
    """I(X;X) = H(X)."""
    assert compute_mutual_information(BALANCED, BALANCED) == pytest.approx(1.0)


def test_mutual_information_is_invariant_to_cluster_relabeling():
    """Cluster ids are arbitrary labels."""
    relabeled = {node: 7 - cluster for node, cluster in BALANCED.items()}
    assert compute_mutual_information(BALANCED, relabeled) == pytest.approx(1.0)


def test_mutual_information_is_zero_for_independent_partitions():
    """{0,1}|{2,3} against {0,2}|{1,3}: every joint cell has p_ij = p_i p_j."""
    independent = {0: 0, 1: 1, 2: 0, 3: 1}
    assert compute_mutual_information(BALANCED, independent) == pytest.approx(0.0)


def test_mutual_information_against_trivial_partition_is_zero():
    """Knowing a single-cluster partition tells nothing."""
    assert compute_mutual_information(BALANCED, TRIVIAL) == pytest.approx(0.0)


def test_mutual_information_is_symmetric_and_bounded():
    """I(X;Y) = I(Y;X) and I(X;Y) <= min(H(X), H(Y)) on random partitions."""
    for seed in range(5):
        p1, p2 = _random_partitions(seed)
        mi = compute_mutual_information(p1, p2)
        assert mi == pytest.approx(compute_mutual_information(p2, p1))
        bound = min(compute_partition_entropy(p1), compute_partition_entropy(p2))
        assert -1e-12 <= mi <= bound + 1e-12


def test_mutual_information_matches_sklearn_in_bits():
    """Same value as scikit-learn's (natural-log) mutual information converted to bits."""
    for seed in range(5):
        p1, p2 = _random_partitions(seed)
        expected = mutual_info_score(_labels(p1), _labels(p2)) / math.log(2)
        assert compute_mutual_information(p1, p2) == pytest.approx(expected)


def test_mutual_information_rejects_different_node_sets():
    """The two partitions must be over exactly the same nodes."""
    with pytest.raises(ValueError, match="same node set"):
        compute_mutual_information(BALANCED, {0: 0, 1: 0, 2: 1})


def test_mutual_information_of_empty_partitions_is_zero():
    """Two empty partitions compare as zero rather than dividing by zero."""
    assert compute_mutual_information({}, {}) == 0.0


# --- normalized mutual information -----------------------------------------------------


def test_nmi_is_one_for_identical_partitions_under_every_normalization():
    """Every normalization gives 1 when the partitions coincide."""
    for method in ("arithmetic", "geometric", "min", "max"):
        assert compute_normalized_mutual_information(BALANCED, BALANCED, method) == pytest.approx(
            1.0
        )


@pytest.mark.parametrize("method", ["arithmetic", "geometric", "min", "max"])
def test_nmi_matches_sklearn(method: str):
    """Each normalization agrees with scikit-learn's ``average_method`` of the same name."""
    for seed in range(5):
        p1, p2 = _random_partitions(seed)
        expected = normalized_mutual_info_score(_labels(p1), _labels(p2), average_method=method)
        assert compute_normalized_mutual_information(p1, p2, method) == pytest.approx(expected)


def test_nmi_edge_cases_with_trivial_partitions():
    """Two trivial partitions count as identical; one trivial partition scores zero."""
    assert compute_normalized_mutual_information(TRIVIAL, TRIVIAL) == 1.0
    assert compute_normalized_mutual_information(BALANCED, TRIVIAL) == 0.0
    assert compute_normalized_mutual_information(TRIVIAL, BALANCED) == 0.0


def test_nmi_rejects_unknown_method():
    """An unknown normalization is a usage error."""
    with pytest.raises(ValueError, match="Unknown normalization method"):
        compute_normalized_mutual_information(BALANCED, BALANCED, method="harmonic")


# --- variation of information ----------------------------------------------------------


def test_variation_of_information_is_zero_for_identical_partitions():
    """VI is a distance: zero between a partition and itself."""
    assert compute_variation_of_information(BALANCED, BALANCED) == pytest.approx(0.0)


def test_variation_of_information_hand_computed():
    """Balanced vs. singletons: H = 1 and 2, MI = 1, so VI = 1 + 2 - 2 = 1."""
    assert compute_variation_of_information(BALANCED, SINGLETONS) == pytest.approx(1.0)


def test_variation_of_information_is_a_metric():
    """Symmetric, non-negative and satisfying the triangle inequality on random partitions."""
    for seed in range(5):
        p1, p2 = _random_partitions(seed)
        p3, _ = _random_partitions(seed + 100)
        d12 = compute_variation_of_information(p1, p2)
        d21 = compute_variation_of_information(p2, p1)
        d13 = compute_variation_of_information(p1, p3)
        d32 = compute_variation_of_information(p3, p2)
        assert d12 == pytest.approx(d21)
        assert d12 >= -1e-12
        assert d12 <= d13 + d32 + 1e-9


# --- adjusted Rand index ---------------------------------------------------------------


def test_ari_is_one_for_identical_partitions_and_relabelings():
    """ARI ignores the cluster labels."""
    relabeled = {node: cluster + 10 for node, cluster in BALANCED.items()}
    assert compute_adjusted_rand_index(BALANCED, BALANCED) == pytest.approx(1.0)
    assert compute_adjusted_rand_index(BALANCED, relabeled) == pytest.approx(1.0)


def test_ari_of_crossed_partitions_is_negative():
    """{0,1}|{2,3} vs {0,2}|{1,3} agree on no pair: ARI = -0.5, below chance level (0)."""
    crossed = {0: 0, 1: 1, 2: 0, 3: 1}
    assert compute_adjusted_rand_index(BALANCED, crossed) == pytest.approx(-0.5)


def test_ari_rejects_different_node_sets():
    """The two partitions must be over exactly the same nodes."""
    with pytest.raises(ValueError, match="same node set"):
        compute_adjusted_rand_index(BALANCED, {**BALANCED, 4: 0})


# --- comparison helpers ----------------------------------------------------------------


def test_compare_partitions_reports_every_metric():
    """The summary carries NMI, ARI, VI and the cluster counts."""
    result = compare_partitions(BALANCED, SINGLETONS)
    assert set(result) == {"nmi", "ari", "vi", "num_clusters_1", "num_clusters_2"}
    assert result["num_clusters_1"] == 2
    assert result["num_clusters_2"] == 4
    assert result["nmi"] == pytest.approx(
        compute_normalized_mutual_information(BALANCED, SINGLETONS)
    )
    assert result["ari"] == pytest.approx(compute_adjusted_rand_index(BALANCED, SINGLETONS))
    assert result["vi"] == pytest.approx(1.0)


def test_compare_decompositions_treats_shells_as_clusters(karate: nx.Graph):
    """A decomposition against itself is a perfect match; against a coarser one it is not."""
    kcores = compute_kcores(nx.Graph(karate.edges()))  # unweighted: shells 1..4
    same = compare_decompositions(kcores, kcores)
    assert same["nmi"] == pytest.approx(1.0)
    assert same["ari"] == pytest.approx(1.0)
    assert same["vi"] == pytest.approx(0.0)
    assert same["num_clusters_1"] == len(set(kcores.node_indices.values()))

    coarse = kcores.model_copy(deep=True)
    coarse.node_indices = {node: min(index, 2) for node, index in kcores.node_indices.items()}
    different = compare_decompositions(kcores, coarse)
    assert different["nmi"] < 1.0
    assert different["num_clusters_2"] == 2


# --- per-cluster set similarities ------------------------------------------------------


def test_overlap_coefficient_uses_the_smaller_cluster():
    """|A ∩ B| / min(|A|, |B|): a cluster nested in a bigger one overlaps fully."""
    p1 = {0: 0, 1: 0, 2: 1}
    p2 = {0: 0, 1: 0, 2: 0}
    assert compute_overlap_coefficient(p1, p2, 0, 0) == pytest.approx(1.0)
    assert compute_overlap_coefficient(p1, p2, 1, 0) == pytest.approx(1.0)


def test_overlap_coefficient_is_zero_for_disjoint_or_missing_clusters():
    """No shared node, or a cluster id that does not exist, gives 0."""
    p1 = {0: 0, 1: 0, 2: 1}
    p2 = {0: 1, 1: 1, 2: 0}
    assert compute_overlap_coefficient(p1, p2, 0, 0) == 0.0
    assert compute_overlap_coefficient(p1, p2, 0, 99) == 0.0


def test_jaccard_similarity_of_clusters():
    """|A ∩ B| / |A ∪ B| on a hand-built pair of clusters."""
    p1 = {0: 0, 1: 0, 2: 0, 3: 1}
    p2 = {0: 0, 1: 0, 2: 1, 3: 1}
    assert compute_jaccard_similarity(p1, p2, 0, 0) == pytest.approx(2 / 3)
    assert compute_jaccard_similarity(p1, p2, 1, 1) == pytest.approx(1 / 2)
    assert compute_jaccard_similarity(p1, p2, 0, 1) == pytest.approx(1 / 4)


def test_jaccard_similarity_edge_cases():
    """Two missing clusters are identical (1.0); one missing cluster is disjoint (0.0)."""
    p1 = {0: 0}
    assert compute_jaccard_similarity(p1, p1, 5, 6) == 1.0
    assert compute_jaccard_similarity(p1, p1, 0, 6) == 0.0
