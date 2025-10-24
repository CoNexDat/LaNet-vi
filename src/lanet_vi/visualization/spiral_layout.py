"""Spiral layout algorithm for network visualization.

This module implements a mathematical spiral placement algorithm ported from
the original C++ LaNet-vi implementation (espiral.cpp). It uses a Newton-Raphson
iterative solver to compute node positions along a spiral path.
"""

import math
from typing import Dict, Tuple

import numpy as np

from lanet_vi.logging_config import get_logger

logger = get_logger(__name__)


def calcular_rho(
    K: float,
    rho_i: float,
    r_i: float,
    r_j: float,
    sep: float,
    beta: float,
    max_iterations: int = 100,
    tolerance: float = 0.1,
) -> float:
    """
    Calculate the radial position for spiral layout using Newton-Raphson method.

    This function solves the equation: sqrt(x^2 + K1*x*cos(theta) + K2) = sep
    where theta = (x/K)^(1/beta) - (rho_i/K)^(1/beta)

    Direct port from legacy/Source/espiral.cpp

    Parameters
    ----------
    K : float
        Spiral scaling constant
    rho_i : float
        Previous radial position
    r_i : float
        Radius of previous node
    r_j : float
        Radius of current node
    sep : float
        Target separation distance between nodes
    beta : float
        Spiral tightness parameter (controls how quickly spiral expands)
    max_iterations : int
        Maximum Newton-Raphson iterations (default: 100)
    tolerance : float
        Convergence tolerance as fraction of sep (default: 0.1)

    Returns
    -------
    float
        New radial position along spiral

    Notes
    -----
    The algorithm uses Newton-Raphson iteration to solve for the position where
    the distance between consecutive nodes equals the desired separation.

    Examples
    --------
    >>> rho = calcular_rho(K=10.0, rho_i=5.0, r_i=0.5, r_j=0.5, sep=1.0, beta=1.5)
    """
    # Equation coefficients
    K2 = rho_i ** 2
    K1 = -2 * rho_i

    # Initial guess for new position
    # Using spiral formula: K * ((rho_i/K)^(1/beta) + sep/(2*rho_i))^beta
    x_j = K * ((rho_i / K) ** (1 / beta) + sep / (2 * rho_i)) ** beta
    x_i = x_j

    # Newton-Raphson iteration
    def compute_f(x: float) -> float:
        """Compute f(x) = distance - sep."""
        theta = (x / K) ** (1 / beta) - (rho_i / K) ** (1 / beta)
        distance = math.sqrt(x**2 + K1 * x * math.cos(theta) + K2)
        return distance - sep

    def compute_f_derivative(x: float, f_val: float) -> float:
        """Compute derivative f'(x)."""
        theta = (x / K) ** (1 / beta) - (rho_i / K) ** (1 / beta)
        term1 = 2 * x
        term2 = K1 * math.cos(theta)
        term3 = -K1 * (x / K) * (1 / beta) * ((x / K) ** (1 / beta - 1)) * math.sin(theta)

        # Derivative of sqrt term
        distance = f_val + sep
        if distance < 1e-10:
            return 0.0

        return (0.5 / distance) * (term1 + term2 + term3)

    # Iterative solver
    f = compute_f(x_i)
    iterations = 0

    while abs(f) > tolerance * sep and iterations < max_iterations:
        iterations += 1

        f_der = compute_f_derivative(x_i, f)

        if abs(f_der) < 1e-10:
            logger.warning(f"Derivative too small in spiral calculation at iteration {iterations}")
            break

        # Newton-Raphson update
        x_j = x_i - f / f_der

        # Ensure x_j stays positive and reasonable
        if x_j < 0:
            x_j = x_i * 0.5
        elif x_j > 10 * rho_i:
            x_j = 10 * rho_i

        f = compute_f(x_j)
        x_i = x_j

    if iterations >= max_iterations:
        logger.debug(f"Spiral calculation reached max iterations ({max_iterations})")

    return x_i


def compute_spiral_positions(
    num_nodes: int,
    K: float = 10.0,
    beta: float = 1.5,
    separation: float = 1.0,
    initial_radius: float = 1.0,
    node_radius: float = 0.5,
) -> Dict[int, Tuple[float, float]]:
    """
    Compute node positions along a spiral path.

    Parameters
    ----------
    num_nodes : int
        Number of nodes to position
    K : float
        Spiral scaling constant (default: 10.0)
    beta : float
        Spiral tightness (default: 1.5, higher = tighter spiral)
    separation : float
        Target separation between consecutive nodes (default: 1.0)
    initial_radius : float
        Starting radius for the spiral (default: 1.0)
    node_radius : float
        Radius of each node for collision avoidance (default: 0.5)

    Returns
    -------
    Dict[int, Tuple[float, float]]
        Mapping from node index to (x, y) coordinates

    Examples
    --------
    >>> positions = compute_spiral_positions(100, K=15.0, beta=2.0)
    >>> len(positions)
    100
    """
    logger.info(f"Computing spiral layout for {num_nodes} nodes (K={K}, beta={beta}, sep={separation})")

    positions = {}

    if num_nodes == 0:
        return positions

    # First node at initial position
    rho = initial_radius
    theta = 0.0

    positions[0] = (rho * math.cos(theta), rho * math.sin(theta))

    # Compute remaining nodes along spiral
    for i in range(1, num_nodes):
        # Calculate new radial position using Newton-Raphson
        rho = calcular_rho(
            K=K,
            rho_i=rho,
            r_i=node_radius,
            r_j=node_radius,
            sep=separation,
            beta=beta,
        )

        # Calculate angular position from radial position
        # theta follows spiral equation: rho = K * theta^beta
        theta = (rho / K) ** (1 / beta)

        # Convert polar to Cartesian
        x = rho * math.cos(theta)
        y = rho * math.sin(theta)

        positions[i] = (x, y)

    logger.debug(f"Spiral layout complete: radius range {initial_radius:.2f} to {rho:.2f}")

    return positions


def compute_semicircular_positions(
    num_nodes: int,
    radius: float = 10.0,
    start_angle: float = 0.0,
    end_angle: float = np.pi,
) -> Dict[int, Tuple[float, float]]:
    """
    Compute node positions along a semicircular arc.

    This is a simpler alternative to spiral layout for creating
    curved arrangements.

    Parameters
    ----------
    num_nodes : int
        Number of nodes to position
    radius : float
        Radius of the semicircle (default: 10.0)
    start_angle : float
        Starting angle in radians (default: 0.0)
    end_angle : float
        Ending angle in radians (default: π for semicircle)

    Returns
    -------
    Dict[int, Tuple[float, float]]
        Mapping from node index to (x, y) coordinates

    Examples
    --------
    >>> positions = compute_semicircular_positions(50, radius=15.0)
    >>> len(positions)
    50
    """
    logger.info(f"Computing semicircular layout for {num_nodes} nodes (radius={radius})")

    positions = {}

    if num_nodes == 0:
        return positions

    if num_nodes == 1:
        # Single node at start position
        positions[0] = (radius * math.cos(start_angle), radius * math.sin(start_angle))
        return positions

    # Evenly space nodes along arc
    angles = np.linspace(start_angle, end_angle, num_nodes)

    for i, theta in enumerate(angles):
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        positions[i] = (x, y)

    return positions
