"""Circular averaging for angular coordinates."""

import numpy as np


class CircularAverage:
    """
    Circular average calculator for angles.

    This implements the circular mean algorithm used in LaNet-vi C++ code
    to calculate weighted average of angular positions on a circle.

    Based on: legacy/3.0.1/kcores_component.cpp lines 406-425
    """

    def average(
        self,
        current_angle: float,
        current_weight: float,
        new_angle: float,
        new_weight: float
    ) -> float:
        """
        Calculate weighted circular average of two angles.

        Parameters
        ----------
        current_angle : float
            Current average angle (radians)
        current_weight : float
            Weight of current angle
        new_angle : float
            New angle to add (radians)
        new_weight : float
            Weight of new angle

        Returns
        -------
        float
            Updated circular average (radians)

        Notes
        -----
        The circular average is calculated using vector addition:
        1. Convert angles to unit vectors
        2. Weight and sum the vectors
        3. Convert back to angle

        This prevents wrapping issues at 0/2π boundary.
        """
        if current_weight == 0:
            return new_angle

        if new_weight == 0:
            return current_angle

        total_weight = current_weight + new_weight

        # Convert to Cartesian coordinates (unit vectors)
        x_current = current_weight * np.cos(current_angle)
        y_current = current_weight * np.sin(current_angle)

        x_new = new_weight * np.cos(new_angle)
        y_new = new_weight * np.sin(new_angle)

        # Sum weighted vectors
        x_total = x_current + x_new
        y_total = y_current + y_new

        # Convert back to angle
        return np.arctan2(y_total, x_total)


def calculate_phi_from_neighbors(
    node_id: int,
    shell_index: int,
    max_shell_index: int,
    neighbors: list,
    node_shells: dict,
    node_positions: dict,
    component_center: tuple,
    is_weighted: bool = False,
    edge_weights: dict = None,
    no_cliques: bool = False,
    rng: np.random.Generator = None
) -> float:
    """
    Calculate phi (angular position) based on neighbors in higher shells.

    This is the core algorithm from LaNet-vi C++ code (kcores_component.cpp:397-443)
    that creates smooth circular distributions.

    Parameters
    ----------
    node_id : int
        Current node ID
    shell_index : int
        Shell/core index of current node
    max_shell_index : int
        Maximum shell index in graph
    neighbors : list
        List of neighbor node IDs
    node_shells : dict
        Mapping from node ID to shell index
    node_positions : dict
        Current positions of nodes (may be partial during layout)
    component_center : tuple
        (x, y) center of this component
    is_weighted : bool
        Whether graph is weighted
    edge_weights : dict
        Edge weights (if weighted)
    no_cliques : bool
        If True, use Gaussian distribution instead of neighbor-based
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    float
        Angular position (phi) in radians [0, 2π]
    """
    if rng is None:
        rng = np.random.default_rng()

    # Random angular offset (C++ line 398)
    ang_init = 2.0 * np.pi * rng.uniform(0, 1)

    if no_cliques:
        # Simple mode: random distribution with Gaussian noise
        # (C++ line 439 - for no_cliques=true)
        return 2.0 * np.pi * rng.uniform(0, 1)

    # Find neighbors in higher shells (C++ lines 410-426)
    ca = CircularAverage()
    phi_host = 0.0
    amount = 0
    sum_w = 0.0

    # Calculate weights if needed
    if is_weighted and edge_weights:
        for nb in neighbors:
            if node_shells.get(nb, 0) > shell_index:
                edge_key = (min(node_id, nb), max(node_id, nb))
                sum_w += edge_weights.get(edge_key, 1.0)

    # Calculate circular average of neighbor angles
    for nb in neighbors:
        nb_shell = node_shells.get(nb, 0)

        # Only consider neighbors in higher shells (C++ line 411)
        if nb_shell <= shell_index:
            continue

        # Skip if neighbor position not yet calculated
        if nb not in node_positions:
            continue

        # Calculate weight for this neighbor
        if is_weighted and edge_weights:
            edge_key = (min(node_id, nb), max(node_id, nb))
            w = edge_weights.get(edge_key, 1.0)
            new_amount = int(w / sum_w) if sum_w > 0 else 0  # C++ line 418
        else:
            # Weight by shell difference (C++ line 420)
            new_amount = (nb_shell + 1 - shell_index)

        if new_amount == 0:
            continue

        # Get neighbor position relative to component center
        nb_x, nb_y = node_positions[nb]
        cx, cy = component_center

        # Calculate angle to neighbor (C++ line 423)
        neighbor_angle = np.arctan2(nb_y - cy, nb_x - cx) + ang_init

        # Update circular average
        phi_host = ca.average(phi_host, amount, neighbor_angle, new_amount)
        amount += new_amount

    # Remove random offset (C++ line 429)
    phi_host -= ang_init

    # If no neighbors in higher shells, use random angle (C++ lines 433-435)
    if amount == 0:
        phi_host = 2.0 * np.pi * rng.uniform(0, 1)

    # Normalize to [0, 2π]
    while phi_host < 0:
        phi_host += 2.0 * np.pi
    while phi_host >= 2.0 * np.pi:
        phi_host -= 2.0 * np.pi

    return phi_host
