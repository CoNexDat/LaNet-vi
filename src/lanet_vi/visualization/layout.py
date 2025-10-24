"""Layout algorithms for network visualization, including circle packing."""

import logging
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from lanet_vi.models.config import CoordDistributionAlgorithm, LayoutConfig
from lanet_vi.models.graph import Component
from lanet_vi.visualization.circular_average import calculate_phi_from_neighbors

# Set up debug logger for layout analysis
logger = logging.getLogger(__name__)
DEBUG_LAYOUT = False  # Set to True to enable detailed layout logging


def place_in_circular_sector(
    ratio: float,
    alfa: float,
    n: int,
    total: int,
    random: float,
    unique: bool = False,
) -> Tuple[float, float]:
    """
    Place a node within a circular sector using a U-shaped path algorithm.

    This is a direct port of the C++ `placeInCircularSector` function from
    the original LaNet-vi implementation (kcores_component.cpp:48-74).

    The algorithm creates a U-shaped path within an angular sector:
    1. Start at center (rho=0, phi=0)
    2. Move outward to radius `ratio` (increasing rho)
    3. Sweep along arc from 0 to `alfa` (increasing phi)
    4. Move inward back to center (decreasing rho)

    This ensures nodes from the same component cluster together angularly.

    Parameters
    ----------
    ratio : float
        Radial extent (0 to 1), typically 0.9 for component nodes
    alfa : float
        Angular width allocated to this component (in radians)
    n : int
        Node index within component (0-based)
    total : int
        Total number of nodes in component
    random : float
        Random offset for variation (0 to 1)
    unique : bool
        Whether this is the only component in the shell

    Returns
    -------
    Tuple[float, float]
        (rho, phi) - polar coordinates relative to component sector
        rho: radial position (0 to ratio)
        phi: angular position (0 to alfa)
    """
    # Calculate position along U-shaped path (0 to 1)
    pos = (n / total) + random
    if pos > 1.0:
        pos = pos - 1.0

    # Calculate distance along the U-shaped path
    if not unique:
        # Path: out (ratio) + arc (ratio * alfa) + in (ratio)
        divs = pos * (2.0 * ratio + ratio * alfa)
    else:
        # Single component: skip first outward leg
        divs = ratio + pos * ratio * alfa

    # Convert path distance to polar coordinates
    rho = 0.0
    phi = 0.0

    if divs < ratio:
        # Moving outward along first radial segment
        rho = divs
        phi = 0.0
    elif divs < ratio + ratio * alfa:
        # Moving along arc at maximum radius
        rho = ratio
        phi = (divs - ratio) / ratio
    else:
        # Moving inward along final radial segment
        rho = ratio - (divs - ratio - (ratio * alfa))
        phi = alfa

    return (rho, phi)


class SpatialHashGrid:
    """
    Spatial hash grid for efficient circle overlap detection.

    Divides 2D space into grid cells for O(1) neighbor queries.
    """

    def __init__(self, bounds: Tuple[float, float, float, float], cell_size: float):
        """
        Initialize spatial hash grid.

        Parameters
        ----------
        bounds : Tuple[float, float, float, float]
            (min_x, max_x, min_y, max_y) bounds of the space
        cell_size : float
            Size of each grid cell
        """
        self.min_x, self.max_x, self.min_y, self.max_y = bounds
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int], Set[int]] = {}

    def _get_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Get grid cell coordinates for a point."""
        cell_x = int((x - self.min_x) / self.cell_size)
        cell_y = int((y - self.min_y) / self.cell_size)
        return (cell_x, cell_y)

    def insert(self, idx: int, x: float, y: float, radius: float):
        """Insert a circle into the grid."""
        # Find all cells this circle overlaps
        cells = self._get_overlapping_cells(x, y, radius)
        for cell in cells:
            if cell not in self.grid:
                self.grid[cell] = set()
            self.grid[cell].add(idx)

    def remove(self, idx: int, x: float, y: float, radius: float):
        """Remove a circle from the grid."""
        cells = self._get_overlapping_cells(x, y, radius)
        for cell in cells:
            if cell in self.grid:
                self.grid[cell].discard(idx)

    def _get_overlapping_cells(self, x: float, y: float, radius: float) -> List[Tuple[int, int]]:
        """Get all grid cells overlapped by a circle."""
        min_cell_x = int((x - radius - self.min_x) / self.cell_size)
        max_cell_x = int((x + radius - self.min_x) / self.cell_size)
        min_cell_y = int((y - radius - self.min_y) / self.cell_size)
        max_cell_y = int((y + radius - self.min_y) / self.cell_size)

        cells = []
        for cx in range(min_cell_x, max_cell_x + 1):
            for cy in range(min_cell_y, max_cell_y + 1):
                cells.append((cx, cy))
        return cells

    def get_nearby_indices(self, x: float, y: float, radius: float) -> Set[int]:
        """Get indices of circles near this position."""
        cells = self._get_overlapping_cells(x, y, radius)
        nearby = set()
        for cell in cells:
            if cell in self.grid:
                nearby.update(self.grid[cell])
        return nearby


def distribute_components(
    components: List[Component],
    center: Tuple[float, float],
    radius: float,
    config: LayoutConfig,
) -> List[Component]:
    """
    Distribute components within a circular container using circle packing.

    This is a Python port of the C++ distribute_components algorithm.
    Uses iterative force-based relaxation to pack circles without overlap.

    Parameters
    ----------
    components : List[Component]
        Components to distribute
    center : Tuple[float, float]
        Center coordinates of container circle
    radius : float
        Radius of container circle
    config : LayoutConfig
        Layout configuration

    Returns
    -------
    List[Component]
        Components with updated center and radius attributes

    Notes
    -----
    The algorithm:
    1. Assigns each component a radius proportional to its weight
    2. Places components randomly within the container
    3. Iteratively resolves overlaps and boundary violations
    4. Optional: Uses spatial hashing for O(N) overlap detection

    For large networks (>1000 components), spatial hashing provides
    significant speedup by avoiding O(N²) pairwise checks.
    """
    if not components:
        return components

    n = len(components)

    # Use spatial hashing for large component counts
    use_spatial_hashing = config.use_spatial_hashing and n > 100
    x0, y0 = center

    # Normalize component weights
    weights = np.array([comp.size for comp in components])
    weights = weights / weights.sum()

    # Initialize random number generator
    rng = np.random.default_rng(config.seed)

    # Calculate component radii based on weights
    radii = np.zeros(n)
    for i in range(n):
        if config.coord_distribution == CoordDistributionAlgorithm.LOG:
            radii[i] = radius * np.sqrt(
                config.alpha * np.log(1 + weights[i]) ** config.beta
            )
        else:
            radii[i] = radius * np.sqrt(config.alpha * weights[i] ** config.beta)

    # Find largest component (will be placed at center)
    i_max_rad = np.argmax(radii)

    # Initialize random positions
    x = np.zeros(n)
    y = np.zeros(n)

    for i in range(n):
        theta = rng.uniform(0, 2 * np.pi)
        rad = rng.uniform(0, radius - radii[i])
        x[i] = x0 + rad * np.cos(theta)
        y[i] = y0 + rad * np.sin(theta)

    # Initialize spatial hash grid if enabled
    spatial_grid = None
    if use_spatial_hashing:
        # Use cell size = 2 * max(radii) for efficient neighbor queries
        max_radius = np.max(radii)
        cell_size = max(2.0 * max_radius, radius / 20.0)
        bounds = (x0 - radius, x0 + radius, y0 - radius, y0 + radius)
        spatial_grid = SpatialHashGrid(bounds, cell_size)

        # Insert all circles into grid
        for i in range(n):
            spatial_grid.insert(i, x[i], y[i], radii[i])

    # Iterative packing procedure
    increment = 1.01
    factor_corrector = 1.10
    max_tries = 10 * n
    max_iterations = 1000
    iteration = 0

    finished = False
    while not finished and iteration < max_iterations:
        iteration += 1

        # Check and fix components outside container
        for i in range(n):
            distance = np.sqrt((x[i] - x0) ** 2 + (y[i] - y0) ** 2)

            if distance > radius - radii[i]:
                # Try to find new valid position
                if spatial_grid:
                    spatial_grid.remove(i, x[i], y[i], radii[i])

                success = _give_new_random_position(
                    i, n, x, y, radii, x0, y0, radius, max_tries, rng, spatial_grid=spatial_grid
                )

                if spatial_grid and success:
                    spatial_grid.insert(i, x[i], y[i], radii[i])

                if not success:
                    # Can't fit - shrink all radii
                    radii *= factor_corrector
                    finished = True
                    break

        if finished:
            break

        # Check and fix overlapping components
        overlap_found = False
        if use_spatial_hashing and spatial_grid:
            # Use spatial hashing for O(N) overlap detection
            for i in range(n):
                nearby = spatial_grid.get_nearby_indices(x[i], y[i], radii[i])
                for j in nearby:
                    if j <= i:  # Avoid duplicate checks
                        continue

                    distance = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
                    min_distance = radii[i] + radii[j]

                    if distance < min_distance:
                        overlap_found = True
                        # Try to move smaller one
                        move_idx = i if radii[i] < radii[j] else j

                        spatial_grid.remove(move_idx, x[move_idx], y[move_idx], radii[move_idx])

                        success = _give_new_random_position(
                            move_idx,
                            n,
                            x,
                            y,
                            radii,
                            x0,
                            y0,
                            radius,
                            max_tries,
                            rng,
                            exclude_idx=-1,
                            spatial_grid=spatial_grid,
                        )

                        if success:
                            spatial_grid.insert(move_idx, x[move_idx], y[move_idx], radii[move_idx])
                        else:
                            # Can't resolve - shrink radii
                            radii *= factor_corrector
                            finished = True
                            break

                if finished:
                    break
        else:
            # Original O(N²) overlap checking
            for i in range(n):
                for j in range(i + 1, n):
                    distance = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
                    min_distance = radii[i] + radii[j]

                    if distance < min_distance:
                        overlap_found = True
                        # Try to move one of them
                        if radii[i] < radii[j]:
                            move_idx = i
                        else:
                            move_idx = j

                        success = _give_new_random_position(
                            move_idx, n, x, y, radii, x0, y0, radius, max_tries, rng, exclude_idx=-1
                        )

                        if not success:
                            # Can't resolve - shrink radii
                            radii *= factor_corrector
                            finished = True
                            break
                if finished:
                    break

        # If no overlaps or boundary violations, we're done
        if not overlap_found and iteration > 10:
            break

    # Update components with positions and radii
    for i, comp in enumerate(components):
        comp.center = (x[i], y[i])
        comp.radius = radii[i]

    return components


def _give_new_random_position(
    idx: int,
    n: int,
    x: np.ndarray,
    y: np.ndarray,
    radii: np.ndarray,
    x0: float,
    y0: float,
    radius: float,
    max_tries: int,
    rng: np.random.Generator,
    exclude_idx: int = -1,
    spatial_grid: SpatialHashGrid = None,
) -> bool:
    """
    Try to find a valid random position for a component.

    Parameters
    ----------
    idx : int
        Index of component to reposition
    n : int
        Total number of components
    x, y : np.ndarray
        Current positions
    radii : np.ndarray
        Component radii
    x0, y0 : float
        Container center
    radius : float
        Container radius
    max_tries : int
        Maximum repositioning attempts
    rng : np.random.Generator
        Random number generator
    exclude_idx : int
        Index to exclude from overlap checking
    spatial_grid : SpatialHashGrid
        Optional spatial hash grid for faster overlap detection

    Returns
    -------
    bool
        True if valid position found, False otherwise
    """
    for _ in range(max_tries):
        # Generate random position
        theta = rng.uniform(0, 2 * np.pi)
        rad = rng.uniform(0, radius - radii[idx])
        new_x = x0 + rad * np.cos(theta)
        new_y = y0 + rad * np.sin(theta)

        # Check if position is valid (no overlaps, within boundary)
        valid = True

        # Check boundary
        distance_to_center = np.sqrt((new_x - x0) ** 2 + (new_y - y0) ** 2)
        if distance_to_center > radius - radii[idx]:
            valid = False
            continue

        # Check overlaps with other components
        if spatial_grid:
            # Use spatial hashing for faster overlap checks
            nearby = spatial_grid.get_nearby_indices(new_x, new_y, radii[idx])
            for j in nearby:
                if j == idx or j == exclude_idx:
                    continue

                distance = np.sqrt((new_x - x[j]) ** 2 + (new_y - y[j]) ** 2)
                if distance < radii[idx] + radii[j]:
                    valid = False
                    break
        else:
            # Original O(N) overlap checking
            for j in range(n):
                if j == idx or j == exclude_idx:
                    continue

                distance = np.sqrt((new_x - x[j]) ** 2 + (new_y - y[j]) ** 2)
                if distance < radii[idx] + radii[j]:
                    valid = False
                    break

        if valid:
            x[idx] = new_x
            y[idx] = new_y
            return True

    return False


def compute_hierarchical_layout(
    components: List[Component],
    config: LayoutConfig,
    max_shell_or_dense: int,
    all_nodes_by_shell: Optional[Dict[int, List[int]]] = None,
    graph: Optional[nx.Graph] = None,
    node_shells: Optional[Dict[int, int]] = None,
    no_cliques: bool = False,
    epsilon: float = 0.18,
) -> Dict[int, Tuple[float, float]]:
    """
    Compute hierarchical layout with shells/dense levels arranged concentrically.

    This function positions ALL nodes in concentric shells (the classic LaNet-vi onion structure),
    using neighbor-based phi calculation to create smooth circular distributions.

    Parameters
    ----------
    components : List[Component]
        Components with shell or dense indices (for component circles)
    config : LayoutConfig
        Layout configuration
    max_shell_or_dense : int
        Maximum shell or dense index value
    all_nodes_by_shell : Optional[Dict[int, List[int]]]
        All nodes grouped by shell index (from decomposition).
        If provided, positions ALL nodes in shell rings.
        If None, falls back to component-based layout.
    graph : Optional[nx.Graph]
        NetworkX graph for neighbor lookup (required for neighbor-based phi)
    node_shells : Optional[Dict[int, int]]
        Mapping from node ID to shell/dense index (required for neighbor-based phi)
    no_cliques : bool
        If True, use simple random distribution instead of neighbor-based phi
    epsilon : float
        Radial variation parameter (0.0-1.0). Higher values = looser spacing.
        Controls ring thickness. Default 0.18 (tight), 0.35 (loose).

    Returns
    -------
    Dict[int, Tuple[float, float]]
        Mapping from node ID to (x, y) coordinates
    """
    node_positions: Dict[int, Tuple[float, float]] = {}

    # If all_nodes_by_shell provided, use shell-based layout (LaNet-vi style)
    if all_nodes_by_shell:
        # Use random number generator for reproducible but randomized layout
        rng = np.random.Generator(np.random.PCG64(config.seed))

        if DEBUG_LAYOUT:
            logger.info("=" * 80)
            logger.info("LAYOUT DEBUG: Starting hierarchical layout")
            logger.info(f"  Max shell/dense: {max_shell_or_dense}")
            logger.info(f"  Total shells: {len(all_nodes_by_shell)}")
            logger.info(f"  Total components: {len(components)}")
            logger.info(f"  Spiral layout: {config.use_spiral_layout}")
            logger.info(f"  Seed: {config.seed}")
            logger.info("=" * 80)

        # Position all nodes in concentric shell rings
        # TWO-PASS ALGORITHM (like C++ findCoordinatesClassic):
        # Pass 1: Initial positioning (many nodes get random angles if no positioned neighbors)
        # Pass 2: Refine with neighbor-based phi (all neighbors now positioned → smooth rings!)

        for pass_num in [1, 2]:
            if DEBUG_LAYOUT and pass_num == 2:
                logger.info(f"\n{'='*80}")
                logger.info("PASS 2: Refining node positions with neighbor-based phi")
                logger.info(f"{'='*80}\n")

            for shell_index in sorted(all_nodes_by_shell.keys(), reverse=True):
                shell_nodes = all_nodes_by_shell[shell_index]
                n_nodes = len(shell_nodes)

                if n_nodes == 0:
                    continue

                # Calculate shell radius (higher k-core = smaller radius, towards center)
                # Use logarithmic scaling for better visual distribution
                shell_radius = (max_shell_or_dense - shell_index + 1) * 80.0

                if DEBUG_LAYOUT and pass_num == 1:
                    logger.info(
                        f"\nShell {shell_index}: {n_nodes} nodes, radius={shell_radius:.2f}"
                    )

                # Position nodes using neighbor-based phi calculation
                # This creates smooth circular distributions like the C++ reference implementation

                if shell_index == max_shell_or_dense:
                    # Innermost shell - place nodes in tight cluster at center
                    # Use simple circular layout for the core
                    angle_offset = rng.uniform(0, 2 * np.pi)
                    for i, node in enumerate(shell_nodes):
                        if n_nodes == 1:
                            node_positions[node] = (0.0, 0.0)
                        else:
                            angle = 2 * np.pi * i / n_nodes + angle_offset
                            r = shell_radius * 0.2  # Tight cluster
                            node_positions[node] = (r * np.cos(angle), r * np.sin(angle))
                else:
                    # Outer shells - use neighbor-based phi calculation
                    # This is the key algorithm from C++ kcores_component.cpp:397-443

                    if DEBUG_LAYOUT and pass_num == 1:
                        logger.info(
                            "  Using neighbor-based phi calculation (smooth circular distribution)"
                        )

                    # Component center is at origin for shell-based layout
                    component_center = (0.0, 0.0)

                    # Epsilon for radial variation (controls ring thickness/spread)
                    eps = epsilon

                    for i, node in enumerate(shell_nodes):
                        # Calculate phi (angular position) based on neighbors in higher shells
                        # This creates smooth distribution by aligning nodes with their neighbors
                        if graph is not None and node_shells is not None:
                            neighbors = list(graph.neighbors(node))
                            phi = calculate_phi_from_neighbors(
                                node_id=node,
                                shell_index=shell_index,
                                max_shell_index=max_shell_or_dense,
                                neighbors=neighbors,
                                node_shells=node_shells,
                                node_positions=node_positions,  # Positions calculated so far
                                component_center=component_center,
                                is_weighted=(
                                    graph.is_weighted() if hasattr(graph, "is_weighted") else False
                                ),
                                edge_weights=None,  # TODO: Add edge weights if needed
                                no_cliques=no_cliques,
                                rng=rng
                            )

                            # Add angular jitter to prevent clustering when nodes share neighbors
                            # This is especially important for medium-core nodes that share
                            # the same high-core neighbors (causing angular clustering)
                            # Add small index-based offset plus Gaussian noise
                            if n_nodes > 1:
                                # Systematic offset based on position in shell
                                systematic_offset = (2.0 * np.pi * i) / n_nodes
                                # Random Gaussian noise (small, ~5 degrees std dev)
                                noise = rng.normal(0, 0.09)  # ~5 degrees
                                # 15% systematic, plus noise
                                phi += systematic_offset * 0.15 + noise
                        else:
                            # Fallback: uniform random distribution if graph not provided
                            phi = 2.0 * np.pi * rng.uniform(0, 1)

                        # Calculate rho (radial position) with small variation
                        # C++ line 392: rhoHost = ratio * (1.0-eps) + eps * ratio * average
                        average = rng.uniform(0, 1)
                        rho_factor = (1.0 - eps) + eps * average
                        rho = shell_radius * rho_factor

                        # Convert polar to Cartesian coordinates
                        # C++ lines 450-456: x = xCoord + (gamma * u * rho * cos(phi))
                        x = rho * np.cos(phi)
                        y = rho * np.sin(phi)

                        node_positions[node] = (x, y)

                        if DEBUG_LAYOUT and pass_num == 1 and i < 3:
                            logger.info(
                                f"      Node {node}: phi={np.degrees(phi):.1f}°, "
                                f"rho={rho:.2f} → ({x:.2f}, {y:.2f})"
                            )

        # Now compute component centers for border circles (only for large components)
        # Skip circle packing if we have too many components - just use simple placement
        if len(components) > 0:
            levels: Dict[int, List[Component]] = {}
            for comp in components:
                level = comp.shell_index if comp.shell_index is not None else comp.dense_index
                if level not in levels:
                    levels[level] = []
                levels[level].append(comp)

            # Distribute component circles at each level
            for level in sorted(levels.keys(), reverse=True):
                level_comps = levels[level]
                shell_radius = (max_shell_or_dense - level + 1) * 80.0

                # For component centers, just place at shell radius with angular distribution
                # Skip expensive circle packing since nodes are already positioned
                n_comps = len(level_comps)
                for i, comp in enumerate(level_comps):
                    angle = 2 * np.pi * i / n_comps
                    # Set component center and radius for border circle rendering
                    comp.center = (shell_radius * np.cos(angle), shell_radius * np.sin(angle))
                    # Estimate radius from component size
                    comp.radius = 10.0 * np.sqrt(comp.size)

        return node_positions

    # Fallback: Original component-based layout (for backward compatibility)
    levels: Dict[int, List[Component]] = {}
    for comp in components:
        level = comp.shell_index if comp.shell_index is not None else comp.dense_index
        if level not in levels:
            levels[level] = []
        levels[level].append(comp)

    # Distribute components at each level concentrically
    for level in sorted(levels.keys(), reverse=True):
        level_comps = levels[level]

        # Calculate radius for this level (larger index = smaller radius, towards center)
        level_radius = (max_shell_or_dense - level + 1) * 100.0

        # Distribute components at this level
        level_comps = distribute_components(
            level_comps,
            center=(0.0, 0.0),
            radius=level_radius,
            config=config,
        )

        # Assign node positions within each component
        for comp in level_comps:
            if comp.center is None or comp.radius is None:
                continue

            cx, cy = comp.center
            comp_radius = comp.radius

            # Distribute nodes within component circle
            n_nodes = len(comp.nodes)
            for i, node in enumerate(comp.nodes):
                if n_nodes == 1:
                    # Single node at center
                    node_positions[node] = (cx, cy)
                else:
                    # Arrange nodes in circle
                    angle = 2 * np.pi * i / n_nodes
                    r = comp_radius * 0.7  # Leave some margin
                    node_positions[node] = (cx + r * np.cos(angle), cy + r * np.sin(angle))

    return node_positions
