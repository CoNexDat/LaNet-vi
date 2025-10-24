"""Spatial indexing utilities for efficient node placement.

This module provides spatial indexing structures for optimizing node placement
algorithms, particularly for large graphs. Uses scipy's cKDTree for efficient
nearest neighbor queries.
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree

from lanet_vi.logging_config import get_logger

logger = get_logger(__name__)


class SpatialIndex:
    """Spatial index for 2D point queries using KD-tree.

    This class provides efficient nearest neighbor and range queries for
    2D points, useful for collision detection and placement optimization
    in graph layouts.

    Parameters
    ----------
    points : Dict[int, Tuple[float, float]]
        Mapping from node ID to (x, y) coordinates

    Attributes
    ----------
    tree : cKDTree
        KD-tree for spatial queries
    node_ids : np.ndarray
        Array of node IDs corresponding to tree points

    Examples
    --------
    >>> positions = {0: (0.0, 0.0), 1: (1.0, 1.0), 2: (2.0, 0.5)}
    >>> index = SpatialIndex(positions)
    >>> neighbors = index.query_radius(x=0.5, y=0.5, radius=1.0)
    """

    def __init__(self, points: Dict[int, Tuple[float, float]]):
        """Initialize spatial index from node positions."""
        if not points:
            raise ValueError("Cannot create spatial index from empty points dictionary")

        # Extract node IDs and coordinates
        self.node_ids = np.array(list(points.keys()))
        coords = np.array([points[nid] for nid in self.node_ids])

        # Build KD-tree
        logger.debug(f"Building KD-tree spatial index for {len(points)} points")
        self.tree = cKDTree(coords)
        logger.debug("Spatial index built successfully")

    def query_nearest(
        self,
        x: float,
        y: float,
        k: int = 1,
        exclude_self: bool = False,
    ) -> List[int]:
        """Find k nearest neighbors to a query point.

        Parameters
        ----------
        x : float
            Query point x-coordinate
        y : float
            Query point y-coordinate
        k : int
            Number of nearest neighbors to return (default: 1)
        exclude_self : bool
            If True, exclude exact matches (default: False)

        Returns
        -------
        List[int]
            List of node IDs of k nearest neighbors

        Examples
        --------
        >>> nearest = index.query_nearest(x=0.5, y=0.5, k=3)
        """
        query_point = np.array([x, y])

        # Query for k+1 neighbors if excluding self (in case of exact match)
        k_query = k + 1 if exclude_self else k

        distances, indices = self.tree.query(query_point, k=k_query)

        # Handle single vs multiple neighbors
        if k_query == 1:
            indices = [indices]
            distances = [distances]

        # Filter out exact matches if requested
        if exclude_self:
            filtered = [
                idx for idx, dist in zip(indices, distances)
                if dist > 1e-10  # Not an exact match
            ]
            indices = filtered[:k]
        else:
            indices = indices[:k] if k_query > 1 else indices

        return [int(self.node_ids[idx]) for idx in indices]

    def query_radius(
        self,
        x: float,
        y: float,
        radius: float,
        exclude_self: bool = False,
    ) -> List[int]:
        """Find all neighbors within a radius of a query point.

        Parameters
        ----------
        x : float
            Query point x-coordinate
        y : float
            Query point y-coordinate
        radius : float
            Search radius
        exclude_self : bool
            If True, exclude exact matches (default: False)

        Returns
        -------
        List[int]
            List of node IDs within radius

        Examples
        --------
        >>> neighbors = index.query_radius(x=1.0, y=1.0, radius=2.0)
        """
        query_point = np.array([x, y])

        indices = self.tree.query_ball_point(query_point, radius)

        # Filter out exact matches if requested
        if exclude_self:
            coords = self.tree.data[indices]
            distances = np.linalg.norm(coords - query_point, axis=1)
            indices = [idx for idx, dist in zip(indices, distances) if dist > 1e-10]

        return [int(self.node_ids[idx]) for idx in indices]

    def query_pairs(self, radius: float) -> List[Tuple[int, int]]:
        """Find all pairs of points within a given radius of each other.

        Parameters
        ----------
        radius : float
            Maximum distance between pairs

        Returns
        -------
        List[Tuple[int, int]]
            List of node ID pairs within radius

        Examples
        --------
        >>> pairs = index.query_pairs(radius=1.5)
        >>> # Returns: [(0, 1), (1, 2), ...]

        Notes
        -----
        Useful for collision detection and clustering analysis.
        """
        # Get sparse distance matrix
        pair_indices = self.tree.query_pairs(radius, output_type='ndarray')

        # Convert indices to node IDs
        pairs = [
            (int(self.node_ids[i]), int(self.node_ids[j]))
            for i, j in pair_indices
        ]

        return pairs

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """Get bounding box of all indexed points.

        Returns
        -------
        Tuple[float, float, float, float]
            (min_x, min_y, max_x, max_y)

        Examples
        --------
        >>> min_x, min_y, max_x, max_y = index.get_bounding_box()
        """
        coords = self.tree.data
        min_x, min_y = coords.min(axis=0)
        max_x, max_y = coords.max(axis=0)

        return float(min_x), float(min_y), float(max_x), float(max_y)


def build_spatial_index(
    node_positions: Dict[int, Tuple[float, float]],
) -> SpatialIndex:
    """Build spatial index from node positions.

    Convenience function for creating a SpatialIndex.

    Parameters
    ----------
    node_positions : Dict[int, Tuple[float, float]]
        Mapping from node ID to (x, y) coordinates

    Returns
    -------
    SpatialIndex
        Spatial index for efficient queries

    Examples
    --------
    >>> positions = {0: (0.0, 0.0), 1: (1.0, 1.0)}
    >>> index = build_spatial_index(positions)
    """
    logger.info(f"Building spatial index for {len(node_positions)} nodes")
    return SpatialIndex(node_positions)


def detect_overlaps(
    node_positions: Dict[int, Tuple[float, float]],
    node_radii: Dict[int, float],
    padding: float = 0.0,
) -> List[Tuple[int, int]]:
    """Detect overlapping nodes using spatial indexing.

    Parameters
    ----------
    node_positions : Dict[int, Tuple[float, float]]
        Node (x, y) positions
    node_radii : Dict[int, float]
        Node radii
    padding : float
        Extra padding to consider as overlap (default: 0.0)

    Returns
    -------
    List[Tuple[int, int]]
        List of overlapping node pairs

    Examples
    --------
    >>> positions = {0: (0.0, 0.0), 1: (0.5, 0.0), 2: (5.0, 0.0)}
    >>> radii = {0: 0.3, 1: 0.3, 2: 0.3}
    >>> overlaps = detect_overlaps(positions, radii, padding=0.1)
    >>> # Returns: [(0, 1)] since nodes 0 and 1 overlap
    """
    logger.debug(f"Detecting overlaps for {len(node_positions)} nodes")

    index = SpatialIndex(node_positions)
    overlaps = []

    for node_id, (x, y) in node_positions.items():
        radius = node_radii.get(node_id, 0.0)

        # Query for potential overlaps
        # Maximum distance for overlap is sum of radii plus padding
        max_radius = radius + max(node_radii.values()) + padding

        nearby = index.query_radius(x, y, max_radius, exclude_self=True)

        # Check actual overlaps
        for other_id in nearby:
            if other_id <= node_id:  # Avoid duplicate pairs
                continue

            other_x, other_y = node_positions[other_id]
            other_radius = node_radii.get(other_id, 0.0)

            # Calculate distance
            dist = np.sqrt((x - other_x)**2 + (y - other_y)**2)

            # Check if overlap
            min_dist = radius + other_radius + padding
            if dist < min_dist:
                overlaps.append((node_id, other_id))

    logger.debug(f"Found {len(overlaps)} overlapping node pairs")
    return overlaps
