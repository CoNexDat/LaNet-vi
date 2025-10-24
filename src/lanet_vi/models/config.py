"""Configuration models for LaNet-vi using Pydantic."""

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class BackgroundColor(str, Enum):
    """Background color options."""

    WHITE = "white"
    BLACK = "black"


class ColorScheme(str, Enum):
    """Color scheme options for visualization."""

    COLOR = "col"
    GRAYSCALE = "bw"
    GRAYSCALE_INTERLACED = "bwi"


class DecompositionType(str, Enum):
    """Type of network decomposition."""

    KCORES = "kcores"
    KDENSES = "kdenses"
    DCORES = "dcores"


class StrengthIntervalMethod(str, Enum):
    """Method for building strength intervals in weighted graphs."""

    EQUAL_NODES = "equalNodesPerInterval"
    EQUAL_SIZE = "equalIntervalSize"
    EQUAL_LOG_SIZE = "equalLogIntervalSize"


class CoordDistributionAlgorithm(str, Enum):
    """Algorithm for distributing components."""

    CLASSIC = "classic"
    POWER = "pow"
    LOG = "log"


class Renderer(str, Enum):
    """Rendering engine options."""

    MATPLOTLIB = "matplotlib"
    NETWORKX = "networkx"
    PLOTLY = "plotly"


class MeasureType(str, Enum):
    """Centrality measure type."""

    KDENSE = "kdense"
    MCORE = "mcore"


class CommunityConfig(BaseModel):
    """Configuration for community detection.

    Parameters
    ----------
    detect_communities : bool
        Whether to detect and visualize communities
    algorithm : Literal["louvain", "greedy_modularity"]
        Community detection algorithm to use
    resolution : float
        Resolution parameter for Louvain algorithm (higher = more communities)
    color_by_community : bool
        Color nodes by community membership instead of k-core
    draw_boundaries : bool
        Draw convex hull boundaries around communities
    draw_circles : bool
        Draw circles around communities
    boundary_alpha : float
        Transparency of community boundaries
    colormap : str
        Matplotlib colormap for community colors
    """

    detect_communities: bool = Field(default=False)
    algorithm: Literal["louvain", "greedy_modularity"] = Field(default="louvain")
    resolution: float = Field(default=1.0, gt=0.0)
    color_by_community: bool = Field(default=True)
    draw_boundaries: bool = Field(default=True)
    draw_circles: bool = Field(default=False)
    boundary_alpha: float = Field(default=0.2, ge=0.0, le=1.0)
    colormap: str = Field(default="tab20")


class VisualizationConfig(BaseModel):
    """Configuration for network visualization.

    Parameters
    ----------
    background : BackgroundColor
        Background color for the visualization
    color_scheme : ColorScheme
        Color scheme for nodes and edges
    width : int
        Image width in pixels
    height : int
        Image height in pixels
    epsilon : float
        Controls ring overlapping possibility
    delta : float
        Controls distance between components
    gamma : float
        Controls component diameter
    font_zoom : float
        Font size multiplier for node labels
    legend_fontsize : Optional[float]
        Legend font size (auto-scales with diagram size if not set)
    edges_percent : float
        Percentage of visible edges (0.0-1.0)
    min_edges : int
        Minimum number of visible edges
    opacity : float
        Edge opacity (0.0-1.0), primarily for SVG/interactive renderers
    unit_length : float
        Base unit length for scaling
    draw_circles : bool
        Whether to draw component border circles
    show_degree_scale : bool
        Whether to show degree scale in the picture
    color_scale_max_value : Optional[int]
        Maximum value for color scale normalization
    gradient_edges : bool
        Whether to use gradient edge coloring
    edge_alpha : float
        Edge transparency (0.0-1.0)
    min_edge_width : float
        Minimum edge thickness
    max_edge_width : float
        Maximum edge thickness
    show_node_labels : bool
        Whether to show node labels
    label_all_nodes : bool
        If True, label all nodes; if False, use label_kcore_range or label_node_list
    label_kcore_min : Optional[int]
        Minimum k-core for labeling (only if label_all_nodes=False)
    label_kcore_max : Optional[int]
        Maximum k-core for labeling (only if label_all_nodes=False)
    """

    # Changed from WHITE to match CAIDA defaults
    background: BackgroundColor = BackgroundColor.BLACK
    color_scheme: ColorScheme = ColorScheme.COLOR
    width: int = Field(default=2400, gt=0)  # Changed from 800 to match CAIDA defaults
    height: int = Field(default=2400, gt=0)  # Changed from 600 to match CAIDA defaults
    epsilon: float = Field(default=0.40, ge=0.0)  # Changed from 0.18 to match CAIDA tuned defaults
    delta: float = Field(default=1.3, gt=0.0)
    gamma: float = Field(default=1.5, gt=0.0)
    font_zoom: float = Field(default=1.0, gt=0.0)
    legend_fontsize: Optional[float] = Field(default=None, gt=0.0)
    edges_percent: float = Field(default=0.5, ge=0.0, le=1.0)  # Changed from 0.0 to 0.5 (50% edges)
    min_edges: int = Field(default=50000, ge=0)  # Changed from 1000 to match CAIDA defaults
    opacity: float = Field(default=0.2, ge=0.0, le=1.0)
    unit_length: float = Field(default=1.0, gt=0.0)
    draw_circles: bool = False
    show_degree_scale: bool = True
    color_scale_max_value: Optional[int] = Field(default=None, gt=0)
    gradient_edges: bool = Field(default=True)
    # Changed from 0.3 to 0.6 for better visibility
    edge_alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    min_edge_width: float = Field(default=0.08, gt=0.0)  # Changed from 0.05 to match CAIDA defaults
    max_edge_width: float = Field(default=0.3, gt=0.0)  # Changed from 0.2 to match CAIDA defaults
    show_node_labels: bool = Field(default=False)
    label_all_nodes: bool = Field(default=True)
    label_kcore_min: Optional[int] = Field(default=None, ge=1)
    label_kcore_max: Optional[int] = Field(default=None, ge=1)
    node_edge_color: Optional[str] = Field(default=None)
    show_size_legend: bool = Field(default=True)
    # Changed from 1.0 to 0.5 for moderate node sizes
    node_size_scale: float = Field(default=0.5, gt=0.0)

    @field_validator("width", "height")
    @classmethod
    def check_aspect_ratio(cls, v: int, info) -> int:
        """Validate that resolution maintains reasonable aspect ratio."""
        if info.field_name == "height" and "width" in info.data:
            width = info.data["width"]
            aspect = width / v
            if aspect < 0.5 or aspect > 3.0:
                raise ValueError(
                    f"Aspect ratio {aspect:.2f} is unusual. "
                    "Recommended ratio is 4:3 (width/height)"
                )
        return v


class DecompositionConfig(BaseModel):
    """Configuration for network decomposition.

    Parameters
    ----------
    decomp_type : DecompositionType
        Type of decomposition to apply
    measure : MeasureType
        Centrality measure to use
    from_layer : int
        Consider graph induced from this layer upward
    granularity : int
        Number of groups in weighted graphs (-1 for max degree)
    strength_intervals : StrengthIntervalMethod
        Method for building strength intervals
    maximum_strength : Optional[float]
        Upper limit for strength intervals
    no_cliques : bool
        Whether to omit cliques in central core
    """

    decomp_type: DecompositionType = DecompositionType.KCORES
    measure: MeasureType = MeasureType.MCORE
    from_layer: int = Field(default=0, ge=0)
    granularity: int = Field(default=-1, ge=-1)
    strength_intervals: StrengthIntervalMethod = StrengthIntervalMethod.EQUAL_SIZE
    maximum_strength: Optional[float] = Field(default=None, gt=0.0)
    no_cliques: bool = False


class GraphConfig(BaseModel):
    """Configuration for graph construction.

    Parameters
    ----------
    multigraph : bool
        Allow repeated edges
    weighted : bool
        Support edge weights
    directed : bool
        Whether graph is directed
    """

    multigraph: bool = False
    weighted: bool = False
    directed: bool = False


class LayoutConfig(BaseModel):
    """Configuration for layout algorithm.

    Parameters
    ----------
    coord_distribution : CoordDistributionAlgorithm
        Algorithm for distributing components
    alpha : float
        Constant in component ratio formula
    beta : float
        Exponent in component ratio formula
    seed : int
        Random seed for reproducibility
    ratio_constant : Optional[float]
        Manual adjustment for node size ratio
    min_component_size : int
        Minimum component size to visualize (filters small components for performance)
    use_spatial_hashing : bool
        Use spatial hashing for faster circle packing (O(N) instead of O(N²))
    use_spiral_layout : bool
        Use spiral layout algorithm for node placement
    spiral_K : float
        Spiral scaling constant (only used if use_spiral_layout=True)
    spiral_beta : float
        Spiral tightness parameter (only used if use_spiral_layout=True)
    spiral_separation : float
        Target separation between consecutive nodes in spiral (only used if use_spiral_layout=True)
    """

    coord_distribution: CoordDistributionAlgorithm = CoordDistributionAlgorithm.CLASSIC
    alpha: float = Field(default=1.0, gt=0.0)
    beta: float = Field(default=1.0)
    # Changed from 42 to 0 for maximum uniformity (CAIDA default)
    seed: int = Field(default=0, ge=0)
    ratio_constant: Optional[float] = Field(default=None, gt=0.0)
    min_component_size: int = Field(default=10, ge=1)  # Changed from 1 to 10 (CAIDA default)
    use_spatial_hashing: bool = Field(default=True)
    use_spiral_layout: bool = Field(default=False)
    spiral_K: float = Field(default=10.0, gt=0.0)
    spiral_beta: float = Field(default=1.5, gt=0.0)
    spiral_separation: float = Field(default=1.0, gt=0.0)


class LaNetConfig(BaseModel):
    """Complete LaNet-vi configuration.

    Parameters
    ----------
    graph : GraphConfig
        Graph construction settings
    decomposition : DecompositionConfig
        Decomposition algorithm settings
    visualization : VisualizationConfig
        Visualization appearance settings
    layout : LayoutConfig
        Layout algorithm settings
    community : CommunityConfig
        Community detection settings
    renderer : Renderer
        Rendering engine to use
    """

    graph: GraphConfig = Field(default_factory=GraphConfig)
    decomposition: DecompositionConfig = Field(default_factory=DecompositionConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    layout: LayoutConfig = Field(default_factory=LayoutConfig)
    community: CommunityConfig = Field(default_factory=CommunityConfig)
    renderer: Renderer = Renderer.MATPLOTLIB

    class Config:
        """Pydantic model configuration."""

        use_enum_values = True
