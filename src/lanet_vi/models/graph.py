"""Graph data models for LaNet-vi."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field


class NodeData(BaseModel):
    """Node attributes and metadata.

    Parameters
    ----------
    node_id : int
        Unique node identifier
    name : Optional[str]
        Human-readable node name
    color : Optional[Tuple[float, float, float]]
        RGB color values (0.0-1.0 range)
    shell_index : Optional[int]
        k-core shell index
    dense_index : Optional[int]
        k-dense index
    degree : Optional[int]
        Node degree
    strength : Optional[float]
        Node strength (sum of edge weights)
    coordinates : Optional[Tuple[float, float]]
        2D visualization coordinates (x, y)
    """

    node_id: int
    name: Optional[str] = None
    color: Optional[Tuple[float, float, float]] = Field(default=None)
    shell_index: Optional[int] = None
    dense_index: Optional[int] = None
    degree: Optional[int] = None
    strength: Optional[float] = None
    coordinates: Optional[Tuple[float, float]] = None

    class Config:
        """Pydantic model configuration."""

        frozen = False
        arbitrary_types_allowed = True


class EdgeData(BaseModel):
    """Edge attributes and metadata.

    Parameters
    ----------
    source : int
        Source node ID
    target : int
        Target node ID
    weight : float
        Edge weight
    visible : bool
        Whether edge should be rendered
    """

    source: int
    target: int
    weight: float = 1.0
    visible: bool = True

    class Config:
        """Pydantic model configuration."""

        frozen = True


class Component(BaseModel):
    """Connected component in the network.

    Parameters
    ----------
    component_id : int
        Unique component identifier
    nodes : List[int]
        List of node IDs in this component
    shell_index : Optional[int]
        k-core shell index for this component
    dense_index : Optional[int]
        k-dense index for this component
    size : int
        Number of nodes in component
    center : Optional[Tuple[float, float]]
        Component center coordinates
    radius : Optional[float]
        Component radius for visualization
    """

    component_id: int
    nodes: List[int]
    shell_index: Optional[int] = None
    dense_index: Optional[int] = None
    size: int = Field(default=0)
    center: Optional[Tuple[float, float]] = None
    radius: Optional[float] = None

    def __init__(self, **data):
        """Initialize component and compute size."""
        super().__init__(**data)
        if self.size == 0:
            self.size = len(self.nodes)

    class Config:
        """Pydantic model configuration."""

        frozen = False


class DecompositionResult(BaseModel):
    """Results from k-core or k-dense decomposition.

    Parameters
    ----------
    decomp_type : str
        Type of decomposition ('kcores' or 'kdenses')
    node_indices : Dict[int, int]
        Mapping from node ID to shell/dense index
    max_index : int
        Maximum shell/dense index
    min_index : int
        Minimum shell/dense index
    components : List[Component]
        List of connected components at each level
    p_function : Optional[List[float]]
        Strength interval boundaries for weighted graphs
    """

    decomp_type: str
    node_indices: Dict[int, int]
    max_index: int
    min_index: int
    components: List[Component] = Field(default_factory=list)
    p_function: Optional[List[float]] = None

    class Config:
        """Pydantic model configuration."""

        frozen = False


class VisualizationLayout(BaseModel):
    """Complete layout information for visualization.

    Parameters
    ----------
    node_positions : Dict[int, Tuple[float, float]]
        Mapping from node ID to (x, y) coordinates
    node_colors : Dict[int, Tuple[float, float, float]]
        Mapping from node ID to RGB color
    node_sizes : Dict[int, float]
        Mapping from node ID to visual size/radius
    visible_edges : List[Tuple[int, int]]
        List of edges to render (source, target)
    edge_colors : Dict[Tuple[int, int], Tuple[Tuple[float, float, float], Tuple[float, float, float]]]
        Mapping from edge (u,v) to pair of colors (color1, color2) for gradient rendering
    edge_widths : Dict[Tuple[int, int], float]
        Mapping from edge (u,v) to line width
    components : List[Component]
        Component layout information
    bounds : Tuple[float, float, float, float]
        Layout bounds (xmin, xmax, ymin, ymax)
    """

    node_positions: Dict[int, Tuple[float, float]]
    node_colors: Dict[int, Tuple[float, float, float]]
    node_sizes: Dict[int, float]
    visible_edges: List[Tuple[int, int]]
    edge_colors: Dict[Tuple[int, int], Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = Field(default_factory=dict)
    edge_widths: Dict[Tuple[int, int], float] = Field(default_factory=dict)
    components: List[Component]
    bounds: Tuple[float, float, float, float]

    class Config:
        """Pydantic model configuration."""

        frozen = False
        arbitrary_types_allowed = True


class NetworkMetadata(BaseModel):
    """Metadata about the network.

    Parameters
    ----------
    num_nodes : int
        Total number of nodes
    num_edges : int
        Total number of edges
    is_weighted : bool
        Whether graph has edge weights
    is_multigraph : bool
        Whether graph allows parallel edges
    is_directed : bool
        Whether graph is directed
    max_degree : int
        Maximum node degree
    min_degree : int
        Minimum node degree
    avg_degree : float
        Average node degree
    density : float
        Graph density
    """

    num_nodes: int
    num_edges: int
    is_weighted: bool = False
    is_multigraph: bool = False
    is_directed: bool = False
    max_degree: int
    min_degree: int
    avg_degree: float
    density: float

    class Config:
        """Pydantic model configuration."""

        frozen = True
