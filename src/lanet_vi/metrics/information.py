"""Information-theoretic metrics for network analysis.

This module implements mutual information, entropy, and related measures
for comparing partitions, decompositions, and community structures.
"""

from collections import Counter
from typing import Dict

import numpy as np

from lanet_vi.logging_config import get_logger

logger = get_logger(__name__)


def compute_partition_entropy(partition: Dict[int, int]) -> float:
    """Compute Shannon entropy of a partition.

    The entropy H(X) measures the uncertainty in the partition:
    H(X) = -Σ p(x) log₂ p(x)

    Parameters
    ----------
    partition : Dict[int, int]
        Mapping from node ID to cluster/community ID

    Returns
    -------
    float
        Entropy in bits

    Examples
    --------
    >>> partition = {0: 0, 1: 0, 2: 1, 3: 1}  # 2 clusters, balanced
    >>> entropy = compute_partition_entropy(partition)
    >>> entropy  # Should be close to 1.0 bit
    1.0

    Notes
    -----
    - Maximum entropy occurs when all clusters have equal size
    - Minimum entropy (0) occurs when all nodes in one cluster
    - Uses log base 2, so entropy is measured in bits
    """
    if not partition:
        return 0.0

    # Count cluster sizes
    cluster_counts = Counter(partition.values())
    total = len(partition)

    # Compute entropy
    entropy = 0.0
    for count in cluster_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)

    return float(entropy)


def compute_mutual_information(
    partition1: Dict[int, int],
    partition2: Dict[int, int],
) -> float:
    """Compute mutual information between two partitions.

    Mutual information I(X;Y) measures how much knowing one partition
    tells us about the other:
    I(X;Y) = Σᵢ Σⱼ p(i,j) log₂(p(i,j) / (p(i)p(j)))

    Parameters
    ----------
    partition1 : Dict[int, int]
        First partition (node ID -> cluster ID)
    partition2 : Dict[int, int]
        Second partition (node ID -> cluster ID)

    Returns
    -------
    float
        Mutual information in bits

    Raises
    ------
    ValueError
        If partitions have different node sets

    Examples
    --------
    >>> p1 = {0: 0, 1: 0, 2: 1, 3: 1}
    >>> p2 = {0: 0, 1: 0, 2: 1, 3: 1}  # Identical partition
    >>> mi = compute_mutual_information(p1, p2)
    >>> mi  # Should equal entropy of either partition
    1.0

    >>> p2 = {0: 1, 1: 0, 2: 1, 3: 0}  # Completely different
    >>> mi = compute_mutual_information(p1, p2)
    >>> mi  # Should be close to 0
    0.0

    Notes
    -----
    - MI = 0 when partitions are independent
    - MI = H(X) = H(Y) when partitions are identical
    - MI ≤ min(H(X), H(Y))
    """
    # Check that partitions have same nodes
    nodes1 = set(partition1.keys())
    nodes2 = set(partition2.keys())

    if nodes1 != nodes2:
        raise ValueError(
            f"Partitions must have same node set. "
            f"Partition1 has {len(nodes1)} nodes, partition2 has {len(nodes2)} nodes. "
            f"Symmetric difference: {len(nodes1.symmetric_difference(nodes2))} nodes"
        )

    n = len(partition1)
    if n == 0:
        return 0.0

    # Build contingency table
    # contingency[i][j] = number of nodes in cluster i of partition1 and cluster j of partition2
    from collections import defaultdict
    contingency = defaultdict(lambda: defaultdict(int))

    for node in nodes1:
        c1 = partition1[node]
        c2 = partition2[node]
        contingency[c1][c2] += 1

    # Compute marginal distributions
    p1 = Counter(partition1.values())  # p(i)
    p2 = Counter(partition2.values())  # p(j)

    # Compute mutual information
    mi = 0.0
    for c1, row in contingency.items():
        for c2, count in row.items():
            if count > 0:
                p_ij = count / n  # p(i,j)
                p_i = p1[c1] / n  # p(i)
                p_j = p2[c2] / n  # p(j)

                mi += p_ij * np.log2(p_ij / (p_i * p_j))

    return float(mi)


def compute_normalized_mutual_information(
    partition1: Dict[int, int],
    partition2: Dict[int, int],
    method: str = "arithmetic",
) -> float:
    """Compute normalized mutual information (NMI) between two partitions.

    NMI normalizes MI to [0, 1] range by dividing by entropy:
    - NMI = 0: partitions are independent
    - NMI = 1: partitions are identical

    Parameters
    ----------
    partition1 : Dict[int, int]
        First partition
    partition2 : Dict[int, int]
        Second partition
    method : str
        Normalization method (default: "arithmetic")
        - "arithmetic": NMI = 2*I(X;Y) / (H(X) + H(Y))
        - "geometric": NMI = I(X;Y) / sqrt(H(X) * H(Y))
        - "min": NMI = I(X;Y) / min(H(X), H(Y))
        - "max": NMI = I(X;Y) / max(H(X), H(Y))

    Returns
    -------
    float
        Normalized mutual information in [0, 1]

    Examples
    --------
    >>> p1 = {0: 0, 1: 0, 2: 1, 3: 1}
    >>> p2 = {0: 0, 1: 0, 2: 1, 3: 1}  # Identical
    >>> nmi = compute_normalized_mutual_information(p1, p2)
    >>> nmi
    1.0

    >>> p2 = {0: 0, 1: 1, 2: 2, 3: 3}  # Different
    >>> nmi = compute_normalized_mutual_information(p1, p2)
    >>> nmi < 0.5
    True

    Notes
    -----
    The choice of normalization affects the score:
    - Arithmetic mean is most common in literature
    - Geometric mean is symmetric
    - Min/max provide bounds on the score
    """
    # Compute MI and entropies
    mi = compute_mutual_information(partition1, partition2)
    h1 = compute_partition_entropy(partition1)
    h2 = compute_partition_entropy(partition2)

    # Handle edge cases
    if h1 == 0.0 and h2 == 0.0:
        return 1.0  # Both trivial partitions
    if h1 == 0.0 or h2 == 0.0:
        return 0.0  # One trivial partition

    # Normalize by chosen method
    if method == "arithmetic":
        nmi = 2 * mi / (h1 + h2)
    elif method == "geometric":
        nmi = mi / np.sqrt(h1 * h2)
    elif method == "min":
        nmi = mi / min(h1, h2)
    elif method == "max":
        nmi = mi / max(h1, h2)
    else:
        raise ValueError(
            f"Unknown normalization method: {method}. "
            "Choose from: arithmetic, geometric, min, max"
        )

    return float(nmi)


def compute_variation_of_information(
    partition1: Dict[int, int],
    partition2: Dict[int, int],
) -> float:
    """Compute variation of information (VI) between two partitions.

    VI is a metric distance between partitions:
    VI(X,Y) = H(X|Y) + H(Y|X) = H(X) + H(Y) - 2*I(X;Y)

    Parameters
    ----------
    partition1 : Dict[int, int]
        First partition
    partition2 : Dict[int, int]
        Second partition

    Returns
    -------
    float
        Variation of information in bits

    Examples
    --------
    >>> p1 = {0: 0, 1: 0, 2: 1, 3: 1}
    >>> p2 = {0: 0, 1: 0, 2: 1, 3: 1}  # Identical
    >>> vi = compute_variation_of_information(p1, p2)
    >>> vi
    0.0

    Notes
    -----
    - VI = 0 when partitions are identical
    - VI is a proper metric (satisfies triangle inequality)
    - Lower VI means more similar partitions
    """
    mi = compute_mutual_information(partition1, partition2)
    h1 = compute_partition_entropy(partition1)
    h2 = compute_partition_entropy(partition2)

    vi = h1 + h2 - 2 * mi

    return float(vi)
