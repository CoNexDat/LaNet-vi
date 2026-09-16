"""Circle packing of sibling components (the C++ ``distribute_components``).

Not yet wired to the layout: the classic placement lives in ``lanet_layout``; this is the
basis for the ``pow``/``log`` coordinate distributions (#18).
"""

import logging

import numpy as np

from lanet_vi.models.config import CoordDistributionAlgorithm, LayoutConfig
from lanet_vi.models.graph import Component

# Set up debug logger for layout analysis
logger = logging.getLogger(__name__)
DEBUG_LAYOUT = False  # Set to True to enable detailed layout logging


def _component_level(comp: Component) -> int:
    """Return the decomposition level (shell or dense index) of a component, 0 if unset."""
    if comp.shell_index is not None:
        return comp.shell_index
    if comp.dense_index is not None:
        return comp.dense_index
    return 0


class SpatialHashGrid:
    """
    Spatial hash grid for efficient circle overlap detection.

    Divides 2D space into grid cells for O(1) neighbor queries.
    """

    def __init__(self, bounds: tuple[float, float, float, float], cell_size: float):
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
        self.grid: dict[tuple[int, int], set[int]] = {}

    def _get_cell(self, x: float, y: float) -> tuple[int, int]:
        """Get grid cell coordinates for a point."""
        cell_x = int((x - self.min_x) / self.cell_size)
        cell_y = int((y - self.min_y) / self.cell_size)
        return (cell_x, cell_y)

    def insert(self, idx: int, x: float, y: float, radius: float) -> None:
        """Insert a circle into the grid."""
        # Find all cells this circle overlaps
        cells = self._get_overlapping_cells(x, y, radius)
        for cell in cells:
            if cell not in self.grid:
                self.grid[cell] = set()
            self.grid[cell].add(idx)

    def remove(self, idx: int, x: float, y: float, radius: float) -> None:
        """Remove a circle from the grid."""
        cells = self._get_overlapping_cells(x, y, radius)
        for cell in cells:
            if cell in self.grid:
                self.grid[cell].discard(idx)

    def _get_overlapping_cells(self, x: float, y: float, radius: float) -> list[tuple[int, int]]:
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

    def get_nearby_indices(self, x: float, y: float, radius: float) -> set[int]:
        """Get indices of circles near this position."""
        cells = self._get_overlapping_cells(x, y, radius)
        nearby = set()
        for cell in cells:
            if cell in self.grid:
                nearby.update(self.grid[cell])
        return nearby


def distribute_components(
    components: list[Component],
    center: tuple[float, float],
    radius: float,
    config: LayoutConfig,
) -> list[Component]:
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
            radii[i] = radius * np.sqrt(config.alpha * np.log(1 + weights[i]) ** config.beta)
        else:
            radii[i] = radius * np.sqrt(config.alpha * weights[i] ** config.beta)

    # Find largest component (will be placed at center)
    _i_max_rad = np.argmax(radii)  # Reserved for future optimization

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
        max_radius = float(np.max(radii))
        cell_size = max(2.0 * max_radius, radius / 20.0)
        bounds = (x0 - radius, x0 + radius, y0 - radius, y0 + radius)
        spatial_grid = SpatialHashGrid(bounds, cell_size)

        # Insert all circles into grid
        for i in range(n):
            spatial_grid.insert(i, x[i], y[i], radii[i])

    # Iterative packing procedure
    _increment = 1.01  # Reserved for future algorithm refinement
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
    spatial_grid: SpatialHashGrid | None = None,
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
            nearby = spatial_grid.get_nearby_indices(new_x, new_y, float(radii[idx]))
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
