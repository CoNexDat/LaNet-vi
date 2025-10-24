"""Main Network class for LaNet-vi."""

import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import networkx as nx

from lanet_vi.decomposition.dcores import compute_dcores, find_components_by_dcore
from lanet_vi.decomposition.kcores import compute_kcores, find_components_by_shell
from lanet_vi.decomposition.kdenses import compute_kdenses, find_components_by_dense
from lanet_vi.io.readers import read_edge_list, read_node_colors, read_node_names
from lanet_vi.logging_config import get_logger
from lanet_vi.models.config import (
    DecompositionType,
    LaNetConfig,
)
from lanet_vi.models.graph import DecompositionResult, VisualizationLayout
from lanet_vi.visualization.colors import compute_shell_color
from lanet_vi.visualization.layout import compute_hierarchical_layout
from lanet_vi.visualization.matplotlib_renderer import render_network, select_visible_edges

logger = get_logger(__name__)


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

    def __init__(self, graph: nx.Graph, config: Optional[LaNetConfig] = None):
        """Initialize Network with graph and configuration."""
        self.graph = graph
        self.config = config if config else LaNetConfig()
        self.decomposition: Optional[DecompositionResult] = None
        self.node_names: Dict[int, str] = {}
        self.node_colors: Dict[int, Tuple[float, float, float]] = {}

    @classmethod
    def from_edge_list(
        cls,
        file_path: Union[Path, str],
        config: Optional[LaNetConfig] = None,
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

    def load_node_names(self, file_path: Union[Path, str]) -> None:
        """
        Load node names from file.

        Parameters
        ----------
        file_path : Union[Path, str]
            Path to node names file
        """
        self.node_names = read_node_names(file_path)

    def load_node_colors(self, file_path: Union[Path, str]) -> None:
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
        decomp_type: Optional[DecompositionType] = None,
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
            self.decomposition = compute_kcores(self.graph, self.config.decomposition)
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
        start_time = time.time()

        # Filter components by minimum size (for component circles only, not node positioning)
        # This only affects which components get border circles drawn, all nodes are still
        # positioned
        min_size = self.config.layout.min_component_size
        filtered_components = [
            comp for comp in self.decomposition.components if comp.size >= min_size
        ]


        # Group ALL nodes by shell index for shell-based layout
        # This ensures every node gets positioned, not just nodes in large components
        from collections import defaultdict

        all_nodes_by_shell = defaultdict(list)
        for node, shell_idx in self.decomposition.node_indices.items():
            all_nodes_by_shell[shell_idx].append(node)

        # Compute hierarchical positions for ALL nodes
        node_positions = compute_hierarchical_layout(
            filtered_components,  # Components for border circles
            self.config.layout,
            self.decomposition.max_index,
            all_nodes_by_shell=dict(all_nodes_by_shell),  # All nodes for positioning
            graph=self.graph,  # For neighbor lookup
            node_shells=self.decomposition.node_indices,  # For shell-based neighbor filtering
            no_cliques=self.config.decomposition.no_cliques,  # Algorithm mode
            epsilon=self.config.visualization.epsilon,  # Radial spread control
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

        # Compute node sizes (proportional to degree, using log scale for large graphs)
        degrees = dict(self.graph.degree())
        max_degree = max(degrees.values()) if degrees else 1
        n_nodes = self.graph.number_of_nodes()

        # Use logarithmic scaling for large networks to keep nodes visible
        scale = self.config.visualization.node_size_scale
        if n_nodes > 1000:
            import math

            node_sizes = {
                node: scale * (1.0 + 8.0 * math.log(1 + degrees[node]) / math.log(1 + max_degree))
                for node in self.graph.nodes()
            }
        else:
            node_sizes = {
                node: scale * (3.0 + 10.0 * (degrees[node] / max_degree))
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

                edge_color_u = tuple(c * 0.75 for c in color_u)
                edge_color_v = tuple(c * 0.75 for c in color_v)

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

                width = (
                    self.config.visualization.min_edge_width
                    + normalized
                    * (
                        self.config.visualization.max_edge_width
                        - self.config.visualization.min_edge_width
                    )
                )
                edge_widths[(u, v)] = width

        # Compute bounds
        if node_positions:
            xs = [pos[0] for pos in node_positions.values()]
            ys = [pos[1] for pos in node_positions.values()]
            bounds = (min(xs), max(xs), min(ys), max(ys))
        else:
            bounds = (0.0, 1.0, 0.0, 1.0)

        return VisualizationLayout(
            node_positions=node_positions,
            node_colors=node_colors,
            node_sizes=node_sizes,
            visible_edges=visible_edges,
            edge_colors=edge_colors,
            edge_widths=edge_widths,
            components=filtered_components,  # Only include components >= min_component_size
            bounds=bounds,
        )

    def visualize(
        self,
        output_path: Union[Path, str],
        layout: Optional[VisualizationLayout] = None,
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

    def get_metadata(self) -> Dict:
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
