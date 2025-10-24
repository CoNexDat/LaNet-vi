"""Community detection algorithms for network analysis.

This module provides community detection functionality including:
- Louvain algorithm for modularity optimization
- Community-based visualization and coloring
"""

from lanet_vi.community.base import Community, CommunityResult
from lanet_vi.community.louvain import detect_communities_louvain

__all__ = [
    "Community",
    "CommunityResult",
    "detect_communities_louvain",
]
