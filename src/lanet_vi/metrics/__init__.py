"""Network metrics and information theory measures.

This module provides various metrics for analyzing network structure,
including mutual information, entropy, and similarity measures.
"""

from lanet_vi.metrics.information import (
    compute_mutual_information,
    compute_normalized_mutual_information,
    compute_partition_entropy,
)
from lanet_vi.metrics.similarity import (
    compare_decompositions,
    compare_partitions,
    compute_adjusted_rand_index,
)

__all__ = [
    "compute_mutual_information",
    "compute_normalized_mutual_information",
    "compute_partition_entropy",
    "compare_partitions",
    "compare_decompositions",
    "compute_adjusted_rand_index",
]
