"""Similarity metrics for comparing partitions and decompositions.

This module provides metrics for comparing different partitions, clusterings,
and decompositions of networks.
"""

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import adjusted_rand_score

from lanet_vi.logging_config import get_logger
from lanet_vi.metrics.information import (
    compute_normalized_mutual_information,
    compute_variation_of_information,
)
from lanet_vi.models.graph import DecompositionResult

logger = get_logger(__name__)


def compute_adjusted_rand_index(
    partition1: Dict[int, int],
    partition2: Dict[int, int],
) -> float:
    """Compute Adjusted Rand Index (ARI) between two partitions.

    ARI measures similarity between two clusterings, adjusted for chance:
    - ARI = 1: partitions are identical
    - ARI = 0: partitions are independent (random)
    - ARI < 0: worse than random

    Parameters
    ----------
    partition1 : Dict[int, int]
        First partition (node ID -> cluster ID)
    partition2 : Dict[int, int]
        Second partition (node ID -> cluster ID)

    Returns
    -------
    float
        Adjusted Rand Index in [-1, 1]

    Raises
    ------
    ValueError
        If partitions have different node sets

    Examples
    --------
    >>> p1 = {0: 0, 1: 0, 2: 1, 3: 1}
    >>> p2 = {0: 0, 1: 0, 2: 1, 3: 1}  # Identical
    >>> ari = compute_adjusted_rand_index(p1, p2)
    >>> ari
    1.0

    >>> p2 = {0: 0, 1: 1, 2: 0, 3: 1}  # Different
    >>> ari = compute_adjusted_rand_index(p1, p2)
    >>> ari < 1.0
    True

    Notes
    -----
    Uses sklearn's implementation. ARI is widely used for comparing
    clustering algorithms and community detection methods.
    """
    # Check same node sets
    nodes1 = set(partition1.keys())
    nodes2 = set(partition2.keys())

    if nodes1 != nodes2:
        raise ValueError(
            f"Partitions must have same node set. "
            f"Difference: {len(nodes1.symmetric_difference(nodes2))} nodes"
        )

    # Create aligned label arrays
    nodes = sorted(nodes1)
    labels1 = [partition1[node] for node in nodes]
    labels2 = [partition2[node] for node in nodes]

    # Compute ARI using sklearn
    ari = adjusted_rand_score(labels1, labels2)

    return float(ari)


def compare_partitions(
    partition1: Dict[int, int],
    partition2: Dict[int, int],
) -> Dict[str, float]:
    """Compare two partitions using multiple similarity metrics.

    Computes several metrics for comprehensive comparison:
    - Normalized Mutual Information (NMI)
    - Adjusted Rand Index (ARI)
    - Variation of Information (VI)

    Parameters
    ----------
    partition1 : Dict[int, int]
        First partition
    partition2 : Dict[int, int]
        Second partition

    Returns
    -------
    Dict[str, float]
        Dictionary with similarity metrics:
        - "nmi": Normalized Mutual Information [0, 1]
        - "ari": Adjusted Rand Index [-1, 1]
        - "vi": Variation of Information (lower is better)
        - "num_clusters_1": Number of clusters in partition 1
        - "num_clusters_2": Number of clusters in partition 2

    Examples
    --------
    >>> p1 = {0: 0, 1: 0, 2: 1, 3: 1}
    >>> p2 = {0: 0, 1: 0, 2: 1, 3: 1}
    >>> metrics = compare_partitions(p1, p2)
    >>> metrics["nmi"]
    1.0
    >>> metrics["ari"]
    1.0
    """
    logger.debug("Comparing two partitions")

    nmi = compute_normalized_mutual_information(partition1, partition2)
    ari = compute_adjusted_rand_index(partition1, partition2)
    vi = compute_variation_of_information(partition1, partition2)

    num_clusters_1 = len(set(partition1.values()))
    num_clusters_2 = len(set(partition2.values()))

    results = {
        "nmi": nmi,
        "ari": ari,
        "vi": vi,
        "num_clusters_1": num_clusters_1,
        "num_clusters_2": num_clusters_2,
    }

    logger.info(
        f"Partition comparison: NMI={nmi:.3f}, ARI={ari:.3f}, VI={vi:.3f} "
        f"({num_clusters_1} vs {num_clusters_2} clusters)"
    )

    return results


def compare_decompositions(
    decomp1: DecompositionResult,
    decomp2: DecompositionResult,
) -> Dict[str, float]:
    """Compare two network decompositions.

    Compares decompositions as partitions where each k-core/k-dense level
    is treated as a cluster.

    Parameters
    ----------
    decomp1 : DecompositionResult
        First decomposition
    decomp2 : DecompositionResult
        Second decomposition

    Returns
    -------
    Dict[str, float]
        Similarity metrics (see compare_partitions)

    Examples
    --------
    >>> from lanet_vi.decomposition import compute_kcores
    >>> decomp1 = compute_kcores(G1)
    >>> decomp2 = compute_kcores(G2)
    >>> metrics = compare_decompositions(decomp1, decomp2)

    Notes
    -----
    Useful for comparing:
    - k-core vs k-dense on same graph
    - Same decomposition on different graphs
    - Community detection vs structural decomposition
    """
    logger.info(
        f"Comparing decompositions: {decomp1.decomp_type} vs {decomp2.decomp_type}"
    )

    return compare_partitions(decomp1.node_indices, decomp2.node_indices)


def compute_overlap_coefficient(
    partition1: Dict[int, int],
    partition2: Dict[int, int],
    cluster1_id: int,
    cluster2_id: int,
) -> float:
    """Compute overlap coefficient between two specific clusters.

    Overlap coefficient = |A ∩ B| / min(|A|, |B|)

    Parameters
    ----------
    partition1 : Dict[int, int]
        First partition
    partition2 : Dict[int, int]
        Second partition
    cluster1_id : int
        Cluster ID in partition 1
    cluster2_id : int
        Cluster ID in partition 2

    Returns
    -------
    float
        Overlap coefficient in [0, 1]

    Examples
    --------
    >>> p1 = {0: 0, 1: 0, 2: 1, 3: 1}
    >>> p2 = {0: 0, 1: 0, 2: 0, 3: 1}
    >>> overlap = compute_overlap_coefficient(p1, p2, cluster1_id=0, cluster2_id=0)
    >>> overlap  # Nodes {0, 1} in cluster 0 of p1, {0, 1, 2} in cluster 0 of p2
    1.0  # |{0,1} ∩ {0,1,2}| / min(2, 3) = 2/2 = 1.0
    """
    # Get nodes in each cluster
    nodes1 = {node for node, cid in partition1.items() if cid == cluster1_id}
    nodes2 = {node for node, cid in partition2.items() if cid == cluster2_id}

    if not nodes1 or not nodes2:
        return 0.0

    # Compute overlap
    intersection = len(nodes1 & nodes2)
    min_size = min(len(nodes1), len(nodes2))

    return intersection / min_size


def compute_jaccard_similarity(
    partition1: Dict[int, int],
    partition2: Dict[int, int],
    cluster1_id: int,
    cluster2_id: int,
) -> float:
    """Compute Jaccard similarity between two specific clusters.

    Jaccard similarity = |A ∩ B| / |A ∪ B|

    Parameters
    ----------
    partition1 : Dict[int, int]
        First partition
    partition2 : Dict[int, int]
        Second partition
    cluster1_id : int
        Cluster ID in partition 1
    cluster2_id : int
        Cluster ID in partition 2

    Returns
    -------
    float
        Jaccard similarity in [0, 1]

    Examples
    --------
    >>> p1 = {0: 0, 1: 0, 2: 1, 3: 1}
    >>> p2 = {0: 0, 1: 0, 2: 0, 3: 1}
    >>> jaccard = compute_jaccard_similarity(p1, p2, cluster1_id=0, cluster2_id=0)
    """
    # Get nodes in each cluster
    nodes1 = {node for node, cid in partition1.items() if cid == cluster1_id}
    nodes2 = {node for node, cid in partition2.items() if cid == cluster2_id}

    if not nodes1 and not nodes2:
        return 1.0  # Both empty

    # Compute Jaccard
    intersection = len(nodes1 & nodes2)
    union = len(nodes1 | nodes2)

    if union == 0:
        return 0.0

    return intersection / union
