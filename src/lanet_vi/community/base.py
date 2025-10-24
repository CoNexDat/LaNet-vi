"""Base classes for community detection.

This module provides data structures for representing community detection results.
"""

from typing import Dict, List

from pydantic import BaseModel, Field


class Community(BaseModel):
    """Represents a single community in a network.

    Attributes
    ----------
    id : int
        Unique community identifier
    nodes : List[int]
        List of node IDs in this community
    size : int
        Number of nodes in the community
    """

    id: int
    nodes: List[int]
    size: int = Field(default=0)

    def __init__(self, **data):
        """Initialize community and compute size if not provided."""
        super().__init__(**data)
        if self.size == 0:
            self.size = len(self.nodes)


class CommunityResult(BaseModel):
    """Results from community detection algorithm.

    Attributes
    ----------
    algorithm : str
        Name of the community detection algorithm used
    communities : List[Community]
        List of detected communities
    node_to_community : Dict[int, int]
        Mapping from node ID to community ID
    num_communities : int
        Total number of communities detected
    modularity : float
        Modularity score of the partition (if applicable)
    """

    algorithm: str
    communities: List[Community]
    node_to_community: Dict[int, int]
    num_communities: int = Field(default=0)
    modularity: float = Field(default=0.0)

    def __init__(self, **data):
        """Initialize community result and compute derived fields."""
        super().__init__(**data)
        if self.num_communities == 0:
            self.num_communities = len(self.communities)

    def get_community(self, community_id: int) -> Community | None:
        """Get a community by its ID.

        Parameters
        ----------
        community_id : int
            Community identifier

        Returns
        -------
        Community | None
            The community if found, None otherwise
        """
        for community in self.communities:
            if community.id == community_id:
                return community
        return None

    def get_node_community(self, node: int) -> int | None:
        """Get the community ID for a given node.

        Parameters
        ----------
        node : int
            Node identifier

        Returns
        -------
        int | None
            Community ID if node is found, None otherwise
        """
        return self.node_to_community.get(node)

    def get_community_sizes(self) -> Dict[int, int]:
        """Get the size of each community.

        Returns
        -------
        Dict[int, int]
            Mapping from community ID to size
        """
        return {comm.id: comm.size for comm in self.communities}
