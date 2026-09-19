"""Configuration models for LaNet-vi using Pydantic."""

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    CUSTOM = "custom"


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

    Attributes
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


#: Deprecated ``VisualizationConfig`` fields and the field each one folds into
DEPRECATED_ALIASES = {"show_size_legend": "show_degree_scale", "edge_alpha": "opacity"}


class VisualizationConfig(BaseModel):
    """Configuration for network visualization.

    Attributes
    ----------
    background : BackgroundColor
        Background color for the visualization
    color_scheme : ColorScheme
        Color scheme for nodes and edges
    width : int
        Image width in pixels
    height : int
        Image height in pixels
    window : tuple of four floats
        ``(hstart, hend, vstart, vend)``, fractions of the full picture to render (the
        C++ ``-window``): ``(0, 1, 0, 1)`` is the whole picture, ``(0, 0.5, 0, 0.5)`` its
        top-left quarter, at the same pixel size
    epsilon : float
        Ring thickness as a fraction of its radius (formula (1) of NIPS 2005)
    delta : float
        Shrink factor of sibling components (formula (5))
    gamma : float
        Component diameter; scales the whole picture (formulas (6)-(7))
    font_zoom : float
        Font size multiplier for node labels
    legend_fontsize : Optional[float]
        Legend font size (auto-scales with diagram size if not set)
    edges_percent : float
        Percentage of visible edges (0.0-1.0)
    min_edges : int
        Minimum number of visible edges
    opacity : float
        Edge opacity (0.0-1.0), the C++ ``-opacity``
    unit_length : float
        Base unit length for scaling
    draw_circles : bool
        Whether to draw component border circles
    show_degree_scale : bool
        Whether to show the degree (node size) legend, as the C++ ``-showDegreeScale``
    show_color_legend : bool
        Whether to show the shell/dense index color legend
    color_scale_max_value : Optional[int]
        Maximum value for color scale normalization
    gradient_edges : bool
        Whether to use gradient edge coloring
    show_node_labels : bool
        Whether to show node labels
    label_all_nodes : bool
        If True, label all nodes; if False, use label_kcore_range or label_node_list
    label_kcore_min : Optional[int]
        Minimum k-core for labeling (only if label_all_nodes=False)
    label_kcore_max : Optional[int]
        Maximum k-core for labeling (only if label_all_nodes=False)
    node_edge_color : Optional[str]
        Border color of the nodes (the C++ drew none: border = node color)
    node_size_scale : float
        Multiplier on the node radius (1.0 = the C++ size)
    edge_alpha : float
        Deprecated alias of ``opacity``
    show_size_legend : bool
        Deprecated alias of ``show_degree_scale``
    """

    # Changed from WHITE to match CAIDA defaults
    background: BackgroundColor = BackgroundColor.BLACK
    color_scheme: ColorScheme = ColorScheme.COLOR
    width: int = Field(default=2400, gt=0)  # Changed from 800 to match CAIDA defaults
    height: int = Field(default=2400, gt=0)  # Changed from 600 to match CAIDA defaults
    window: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
    epsilon: float = Field(default=0.18, ge=0.0, le=1.0)  # C++ default: ring thickness
    delta: float = Field(default=1.3, gt=0.0)
    gamma: float = Field(default=1.5, gt=0.0)
    font_zoom: float = Field(default=1.0, gt=0.0)
    legend_fontsize: float | None = Field(default=None, gt=0.0)
    edges_percent: float = Field(default=0.5, ge=0.0, le=1.0)  # Changed from 0.0 to 0.5 (50% edges)
    min_edges: int = Field(default=50000, ge=0)  # Changed from 1000 to match CAIDA defaults
    opacity: float = Field(default=0.2, ge=0.0, le=1.0)
    unit_length: float = Field(default=1.0, gt=0.0)
    draw_circles: bool = False
    show_degree_scale: bool = True
    show_color_legend: bool = True
    color_scale_max_value: int | None = Field(default=None, ge=0)  # 0: valid m-core number
    gradient_edges: bool = Field(default=True)
    show_node_labels: bool = Field(default=False)
    label_all_nodes: bool = Field(default=True)
    label_kcore_min: int | None = Field(default=None, ge=1)
    label_kcore_max: int | None = Field(default=None, ge=1)
    node_edge_color: str | None = Field(default=None)
    node_size_scale: float = Field(default=1.0, gt=0.0)
    # Deprecated aliases (kept so old YAML files still load; see DEPRECATED_ALIASES)
    show_size_legend: bool = Field(default=True)
    edge_alpha: float = Field(default=0.2, ge=0.0, le=1.0)

    @field_validator("window")
    @classmethod
    def window_is_a_sub_rectangle(
        cls, value: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Require ``0 <= hstart < hend <= 1`` and the same vertically."""
        hstart, hend, vstart, vend = value
        if not (0.0 <= hstart < hend <= 1.0 and 0.0 <= vstart < vend <= 1.0):
            raise ValueError("window must be hstart hend vstart vend with 0 <= start < end <= 1")
        return value

    @model_validator(mode="before")
    @classmethod
    def fold_deprecated_aliases(cls, data: Any) -> Any:
        """Map the deprecated aliases onto their current fields.

        An alias only applies when the current field itself is absent, so a file that
        sets the current field is never overridden by the old one.
        """
        if isinstance(data, dict):
            for alias, field in DEPRECATED_ALIASES.items():
                if alias in data:
                    data = dict(data)
                    data.setdefault(field, data.pop(alias))
        return data


class DecompositionConfig(BaseModel):
    """Configuration for network decomposition.

    Attributes
    ----------
    decomp_type : DecompositionType
        Type of decomposition to apply
    measure : MeasureType
        Name of the k-dense measure in the legend: ``mcore`` labels a k-dense as
        ``k - 2`` (its m-core number, the C++ default) and reads ``color_scale_max_value``
        in the same units; ``kdense`` labels it ``k``
    from_layer : int
        Consider graph induced from this layer upward
    granularity : int
        Number of groups in weighted graphs (-1 for max degree)
    strength_intervals : StrengthIntervalMethod
        Method for building strength intervals
    maximum_strength : Optional[float]
        Upper limit for strength intervals
    strength_intervals_file : Optional[Path]
        Boundaries, one per line, for ``strength_intervals = custom``
    no_cliques : bool
        Whether to omit cliques in central core
    """

    decomp_type: DecompositionType = DecompositionType.KCORES
    measure: MeasureType = MeasureType.MCORE
    from_layer: int = Field(default=0, ge=0)
    granularity: int = Field(default=-1, ge=-1)  # -1: maximum degree; otherwise >= 1
    strength_intervals: StrengthIntervalMethod = StrengthIntervalMethod.EQUAL_SIZE
    maximum_strength: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    strength_intervals_file: Path | None = None
    no_cliques: bool = False

    @field_validator("granularity")
    @classmethod
    def granularity_is_sentinel_or_positive(cls, value: int) -> int:
        """``-1`` means "maximum degree"; any other value must be at least 1."""
        if value == 0:
            raise ValueError("granularity must be -1 (maximum degree) or at least 1")
        return value

    @model_validator(mode="after")
    def custom_intervals_need_a_file(self) -> "DecompositionConfig":
        """``strength_intervals = custom`` is only meaningful with a boundaries file."""
        if (
            self.strength_intervals == StrengthIntervalMethod.CUSTOM
            and self.strength_intervals_file is None
        ):
            raise ValueError(
                "strength_intervals 'custom' needs strength_intervals_file "
                "(--strength-intervals-file)"
            )
        return self


class GraphConfig(BaseModel):
    """Configuration for graph construction.

    Attributes
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

    Attributes
    ----------
    coord_distribution : CoordDistributionAlgorithm
        Placement of the components: ``classic`` (concentric rings, the C++ default) or
        the ``pow`` / ``log`` circle packing of sibling components
    alpha : float
        Constant of the disc area law of the circle packing (``pow`` / ``log`` only; the
        C++ default 0.3)
    beta : float
        Exponent of the disc area law (``pow`` / ``log`` only)
    seed : int
        Random seed for reproducibility
    ratio_constant : Optional[float]
        Node radius factor of the ``pow`` / ``log`` modes (the C++ ``-ratioConstant``);
        ``None`` is the C++ auto-adjustment (1 for k-cores, from the top cores for k-dense)
    min_component_size : int
        Minimum component size to visualize (filters small components for performance)
    use_spatial_hashing : bool
        Deprecated, ignored: the circle packing follows the C++ algorithm
    use_spiral_layout : bool
        Use spiral layout algorithm for node placement
    spiral_k : float
        Spiral scaling constant (only used if use_spiral_layout=True)
    spiral_beta : float
        Spiral tightness parameter (only used if use_spiral_layout=True)
    spiral_separation : float
        Target separation between consecutive nodes in spiral (only used if use_spiral_layout=True)
    """

    coord_distribution: CoordDistributionAlgorithm = CoordDistributionAlgorithm.CLASSIC
    alpha: float = Field(default=0.3, gt=0.0)  # C++ default
    beta: float = Field(default=1.0, gt=0.0)  # the C++ raises to 2 / beta
    # Changed from 42 to 0 for maximum uniformity (CAIDA default)
    seed: int = Field(default=0, ge=0)
    ratio_constant: float | None = Field(default=None, gt=0.0)
    min_component_size: int = Field(default=10, ge=1)  # Changed from 1 to 10 (CAIDA default)
    use_spatial_hashing: bool = Field(default=True)
    use_spiral_layout: bool = Field(default=False)
    spiral_k: float = Field(default=10.0, gt=0.0)
    spiral_beta: float = Field(default=1.5, gt=0.0)
    spiral_separation: float = Field(default=1.0, gt=0.0)


class LaNetConfig(BaseModel):
    """Complete LaNet-vi configuration.

    Attributes
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

    model_config = ConfigDict(use_enum_values=True)
