"""Main Network class for LaNet-vi."""

import time
from pathlib import Path

import networkx as nx

from lanet_vi.decomposition.dcores import compute_dcores, find_components_by_dcore
from lanet_vi.decomposition.kcores import compute_kcores, find_components_by_shell
from lanet_vi.decomposition.kdenses import (
    MIN_DENSE_INDEX,
    compute_kdenses,
    find_components_by_dense,
)
from lanet_vi.io.readers import read_edge_list, read_node_colors, read_node_names
from lanet_vi.logging_config import get_logger
from lanet_vi.models.config import (
    DecompositionType,
    LaNetConfig,
)
from lanet_vi.models.graph import Component, DecompositionResult, VisualizationLayout
from lanet_vi.visualization.colors import compute_shell_color
from lanet_vi.visualization.lanet_layout import (
    LayoutParameters,
    compute_lanet_layout,
    node_radius,
)
from lanet_vi.visualization.matplotlib_renderer import render_network, select_visible_edges

logger = get_logger(__name__)


RGB = tuple[float, float, float]


def _darken(color: RGB, factor: float = 0.75) -> RGB:
    """Scale an RGB colour towards black."""
    return (color[0] * factor, color[1] * factor, color[2] * factor)


class Network:
    """
    Main class for network analysis and visualization.

    This class provides a high-level API for:
    - Loading network data
    - Computing k-core or k-dense decomposition
    - Generating visualizations

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph
    config : LaNetConfig
        Complete configuration

    Attributes
    ----------
    graph : nx.Graph
        The network graph
    config : LaNetConfig
        Configuration settings
    decomposition : Optional[DecompositionResult]
        Decomposition results (None until decompose() is called)
    node_names : Dict[int, str]
        Node name mappings
    node_colors : Dict[int, Tuple[float, float, float]]
        Custom node colors

    Examples
    --------
    >>> config = LaNetConfig()
    >>> G = nx.karate_club_graph()
    >>> net = Network(G, config)
    >>> net.decompose()
    >>> net.visualize("output.png")
    """

    def __init__(self, graph: nx.Graph, config: LaNetConfig | None = None):
        """Initialize Network with graph and configuration."""
        self.graph = graph
        self.config = config if config else LaNetConfig()
        self.decomposition: DecompositionResult | None = None
        self.node_names: dict[int, str] = {}
        self.node_colors: dict[int, tuple[float, float, float]] = {}

    @classmethod
    def from_edge_list(
        cls,
        file_path: Path | str,
        config: LaNetConfig | None = None,
    ) -> "Network":
        """
        Create Network from edge list file.

        Parameters
        ----------
        file_path : Union[Path, str]
            Path to edge list file
        config : Optional[LaNetConfig]
            Configuration (uses defaults if None)

        Returns
        -------
        Network
            Network instance

        Examples
        --------
        >>> net = Network.from_edge_list("network.txt")
        """
        if config is None:
            config = LaNetConfig()

        graph = read_edge_list(
            file_path,
            weighted=config.graph.weighted,
            directed=config.graph.directed,
            multigraph=config.graph.multigraph,
        )

        return cls(graph, config)

    def load_node_names(self, file_path: Path | str) -> None:
        """
        Load node names from file.

        Parameters
        ----------
        file_path : Union[Path, str]
            Path to node names file
        """
        self.node_names = read_node_names(file_path)

    def load_node_colors(self, file_path: Path | str) -> None:
        """
        Load custom node colors from file.

        Parameters
        ----------
        file_path : Union[Path, str]
            Path to node colors file
        """
        self.node_colors = read_node_colors(file_path)

    def decompose(
        self,
        decomp_type: DecompositionType | None = None,
    ) -> DecompositionResult:
        """
        Compute network decomposition.

        Parameters
        ----------
        decomp_type : Optional[DecompositionType]
            Type of decomposition (uses config if None)

        Returns
        -------
        DecompositionResult
            Decomposition results

        Examples
        --------
        >>> result = net.decompose(DecompositionType.KCORES)
        """
        if decomp_type is None:
            decomp_type = self.config.decomposition.decomp_type

        logger.info(f"Starting {decomp_type.value} decomposition")
        start_time = time.time()

        if decomp_type == DecompositionType.KCORES:
            # --weighted forces the strength path; otherwise keep autodetecting so a
            # Network built from an already-weighted graph behaves as before
            self.decomposition = compute_kcores(
                self.graph,
                self.config.decomposition,
                weighted=True if self.config.graph.weighted else None,
            )
            self.decomposition = find_components_by_shell(self.graph, self.decomposition)
        elif decomp_type == DecompositionType.KDENSES:
            self.decomposition = compute_kdenses(self.graph)
            self.decomposition = find_components_by_dense(self.graph, self.decomposition)
        elif decomp_type == DecompositionType.DCORES:
            if not self.graph.is_directed():
                raise ValueError(
                    "D-core decomposition requires a directed graph. "
                    "Use KCORES for undirected graphs."
                )
            self.decomposition = compute_dcores(self.graph, self.config.decomposition)
            self.decomposition = find_components_by_dcore(self.graph, self.decomposition)
        else:
            raise ValueError(f"Unknown decomposition type: {decomp_type}")

        elapsed = time.time() - start_time
        logger.info(
            f"Decomposition complete in {elapsed:.2f}s: "
            f"{self.decomposition.min_index}-{self.decomposition.max_index}, "
            f"{len(self.decomposition.components)} components"
        )

        return self.decomposition

    def compute_layout(self) -> VisualizationLayout:
        """
        Compute visualization layout.

        Returns
        -------
        VisualizationLayout
            Layout with node positions and visual properties

        Raises
        ------
        ValueError
            If decompose() hasn't been called yet
        """
        if self.decomposition is None:
            raise ValueError("Must call decompose() before computing layout")

        logger.info("Computing visualization layout")
        _start_time = time.time()  # Reserved for future profiling

        decomposition = self.decomposition
        vis = self.config.visualization
        # Weighted geometry whenever the decomposition ran on strengths (p-function
        # present: --weighted, or weights autodetected) or the graph is declared weighted
        weighted = bool(self.config.graph.weighted) or decomposition.p_function is not None

        # Edge index: for k-dense the edge's own index (metadata), else min of the endpoints
        edge_indices = decomposition.metadata.get("edge_indices")
        node_index = decomposition.node_indices

        def edge_index(u: int, v: int) -> int:
            if edge_indices is not None:
                return int(edge_indices.get((u, v) if u < v else (v, u), MIN_DENSE_INDEX))
            return min(node_index[u], node_index[v])

        params = LayoutParameters(
            epsilon=vis.epsilon,
            delta=vis.delta,
            gamma=vis.gamma,
            u=vis.unit_length,
            no_cliques=self.config.decomposition.no_cliques,
            weighted=weighted,
        )
        lanet = compute_lanet_layout(
            self.graph, node_index, params, seed=self.config.layout.seed, edge_index=edge_index
        )
        node_positions = lanet.positions

        # Nested components (centres and radii) for the border circles, largest first.
        # As the C++ addComponents, only components with clusters of their own get one.
        min_size = self.config.layout.min_component_size
        is_dense = decomposition.decomp_type == "kdenses"
        filtered_components = []
        for i, comp in enumerate(lanet.root.walk()):
            if comp.size < min_size or not comp.clusters:
                continue
            filtered_components.append(
                Component(
                    component_id=i,
                    nodes=[v for cluster in comp.clusters for v in cluster],
                    shell_index=None if is_dense else comp.index,
                    dense_index=comp.index if is_dense else None,
                    size=comp.size,
                    center=(comp.x, comp.y),
                    radius=comp.ratio * comp.u * vis.gamma,
                )
            )

        # Compute node colors
        node_colors = {}
        for node in self.graph.nodes():
            if node in self.node_colors:
                # Use custom color
                node_colors[node] = self.node_colors[node]
            else:
                # Compute color from shell/dense index
                index = self.decomposition.node_indices.get(node, 1)
                node_colors[node] = compute_shell_color(
                    index,
                    self.decomposition.max_index,
                    self.config.visualization.color_scheme,
                    self.config.visualization.color_scale_max_value,
                )

        # Node radii in layout units, as the C++ computeHostRatio (scaled by node_size_scale)
        degrees = dict(self.graph.degree())
        max_degree = max(degrees.values()) if degrees else 1
        strengths: dict[int, float] = {}
        if weighted:
            for v in self.graph:
                strengths[v] = sum(float(d.get("weight", 1.0)) for d in self.graph[v].values())
        max_strength = max(strengths.values(), default=0.0)
        scale = vis.node_size_scale
        node_sizes = {
            node: scale
            * node_radius(
                degrees[node],
                max_degree,
                strengths.get(node, 0.0),
                max_strength,
                weighted=weighted and not self.graph.is_multigraph(),
            )
            for node in self.graph.nodes()
        }

        # Select visible edges
        visible_edges = select_visible_edges(
            self.graph, self.config.visualization, self.decomposition
        )

        # Compute edge colors and widths for gradient rendering
        edge_colors = {}
        edge_widths = {}

        if self.config.visualization.gradient_edges:
            for u, v in visible_edges:
                # Get endpoint colors and darken them (0.75× like original)
                color_u = node_colors.get(u, (0.7, 0.7, 0.7))
                color_v = node_colors.get(v, (0.7, 0.7, 0.7))

                edge_color_u = _darken(color_u)
                edge_color_v = _darken(color_v)

                # IMPORTANT: Colors are flipped in original implementation!
                # The half near node u gets node v's color (showing where it's going)
                # The half near node v gets node u's color (showing where it came from)
                edge_colors[(u, v)] = (edge_color_v, edge_color_u)  # Flipped!

                # Compute edge width based on min degree (like original)
                degree_u = degrees.get(u, 1)
                degree_v = degrees.get(v, 1)
                min_degree = min(degree_u, degree_v)

                # Normalize to min/max edge width range
                if max_degree > 1:
                    normalized = min_degree / max_degree
                else:
                    normalized = 0.5

                width = self.config.visualization.min_edge_width + normalized * (
                    self.config.visualization.max_edge_width
                    - self.config.visualization.min_edge_width
                )
                edge_widths[(u, v)] = width

        # Bounds: the C++ viewport (svg.cpp addHeaders) is 1.6 x 1.2 times 2 * gamma * u * R
        # around the origin, i.e. half-extents of 1.6 and 1.2 times the network radius, so
        # the network sits in a margin where the legends go; widened if a node falls outside
        frame = lanet.frame if lanet.frame > 0 else 1.0
        xs = [pos[0] for pos in node_positions.values()] or [0.0]
        ys = [pos[1] for pos in node_positions.values()] or [0.0]
        bounds = (
            min(-1.6 * frame, min(xs)),
            max(1.6 * frame, max(xs)),
            min(-1.2 * frame, min(ys)),
            max(1.2 * frame, max(ys)),
        )

        return VisualizationLayout(
            node_positions=node_positions,
            node_colors=node_colors,
            node_sizes=node_sizes,
            visible_edges=visible_edges,
            edge_colors=edge_colors,
            edge_widths=edge_widths,
            components=filtered_components,  # nested components >= min_component_size
            bounds=bounds,
        )

    def visualize(
        self,
        output_path: Path | str,
        layout: VisualizationLayout | None = None,
    ) -> None:
        """
        Generate and save visualization.

        Parameters
        ----------
        output_path : Union[Path, str]
            Output file path
        layout : Optional[VisualizationLayout]
            Pre-computed layout (computes if None)

        Examples
        --------
        >>> net.visualize("network.png")
        """
        if self.decomposition is None:
            raise ValueError("Must call decompose() before visualizing")

        if layout is None:
            layout = self.compute_layout()

        render_network(
            self.graph,
            layout,
            self.decomposition,
            self.config.visualization,
            output_path,
            self.node_names if self.node_names else None,
        )

    def get_metadata(self) -> dict:
        """
        Get network metadata.

        Returns
        -------
        Dict
            Dictionary with network statistics
        """
        degrees = dict(self.graph.degree())

        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "is_directed": self.graph.is_directed(),
            "max_degree": max(degrees.values()) if degrees else 0,
            "min_degree": min(degrees.values()) if degrees else 0,
            "avg_degree": sum(degrees.values()) / len(degrees) if degrees else 0,
            "density": nx.density(self.graph),
        }
