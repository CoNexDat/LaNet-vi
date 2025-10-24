"""Random graph generation using NetworkX.

This module provides wrappers around NetworkX's random graph generators,
optimized for LaNet-vi visualization and analysis.
"""

from typing import Optional

import networkx as nx

from lanet_vi.logging_config import get_logger

logger = get_logger(__name__)


def generate_erdos_renyi(
    n: int,
    p: Optional[float] = None,
    m: Optional[int] = None,
    seed: Optional[int] = None,
    directed: bool = False,
) -> nx.Graph:
    """Generate Erdős-Rényi random graph.

    Creates a random graph using either G(n,p) or G(n,m) model:
    - G(n,p): n nodes, each edge exists with probability p
    - G(n,m): n nodes, exactly m edges

    Parameters
    ----------
    n : int
        Number of nodes
    p : Optional[float]
        Probability of edge creation (for G(n,p) model). Must be in [0,1].
        Either p or m must be specified, but not both.
    m : Optional[int]
        Number of edges (for G(n,m) model).
        Either p or m must be specified, but not both.
    seed : Optional[int]
        Random seed for reproducibility
    directed : bool
        If True, generate directed graph (default: False)

    Returns
    -------
    nx.Graph or nx.DiGraph
        Generated random graph

    Raises
    ------
    ValueError
        If neither p nor m is specified, or if both are specified

    Examples
    --------
    >>> # G(n,p) model: 100 nodes, 10% edge probability
    >>> G = generate_erdos_renyi(n=100, p=0.1, seed=42)
    >>> G.number_of_nodes()
    100

    >>> # G(n,m) model: 100 nodes, exactly 500 edges
    >>> G = generate_erdos_renyi(n=100, m=500, seed=42)
    >>> G.number_of_edges()
    500

    Notes
    -----
    Uses NetworkX's fast_gnp_random_graph and gnm_random_graph functions.
    For large dense graphs, G(n,m) model may be faster.
    """
    if p is None and m is None:
        raise ValueError("Either p or m must be specified")
    if p is not None and m is not None:
        raise ValueError("Cannot specify both p and m")

    if p is not None:
        # G(n,p) model
        logger.info(f"Generating Erdős-Rényi G(n={n}, p={p}) graph (directed={directed})")
        if directed:
            G = nx.fast_gnp_random_graph(n, p, seed=seed, directed=True)
        else:
            G = nx.fast_gnp_random_graph(n, p, seed=seed)
    else:
        # G(n,m) model
        logger.info(f"Generating Erdős-Rényi G(n={n}, m={m}) graph (directed={directed})")
        if directed:
            G = nx.gnm_random_graph(n, m, seed=seed, directed=True)
        else:
            G = nx.gnm_random_graph(n, m, seed=seed)

    logger.info(
        f"Generated graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
        f"directed={G.is_directed()}"
    )
    return G


def generate_barabasi_albert(
    n: int,
    m: int,
    seed: Optional[int] = None,
) -> nx.Graph:
    """Generate Barabási-Albert scale-free network.

    Creates a random graph using preferential attachment. The graph exhibits
    a power-law degree distribution, common in real-world networks.

    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    seed : Optional[int]
        Random seed for reproducibility

    Returns
    -------
    nx.Graph
        Generated scale-free graph

    Examples
    --------
    >>> G = generate_barabasi_albert(n=1000, m=3, seed=42)
    >>> G.number_of_nodes()
    1000

    Notes
    -----
    The Barabási-Albert model produces graphs with:
    - Power-law degree distribution P(k) ~ k^(-γ) where γ ≈ 3
    - High clustering coefficient
    - Short average path length (small-world property)

    This model is useful for simulating social networks, the internet,
    and citation networks.
    """
    logger.info(f"Generating Barabási-Albert graph (n={n}, m={m})")

    G = nx.barabasi_albert_graph(n, m, seed=seed)

    logger.info(
        f"Generated scale-free graph: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )
    return G


def generate_watts_strogatz(
    n: int,
    k: int,
    p: float,
    seed: Optional[int] = None,
) -> nx.Graph:
    """Generate Watts-Strogatz small-world network.

    Creates a random graph with small-world properties: high clustering
    and short average path length.

    Parameters
    ----------
    n : int
        Number of nodes
    k : int
        Each node is connected to k nearest neighbors in ring topology
    p : float
        Probability of rewiring each edge (0 ≤ p ≤ 1)
        - p=0: regular ring lattice
        - p=1: random graph
        - 0<p<1: small-world network
    seed : Optional[int]
        Random seed for reproducibility

    Returns
    -------
    nx.Graph
        Generated small-world graph

    Examples
    --------
    >>> G = generate_watts_strogatz(n=1000, k=6, p=0.3, seed=42)
    >>> G.number_of_nodes()
    1000

    Notes
    -----
    The Watts-Strogatz model interpolates between:
    - Regular lattice (p=0): high clustering, long path length
    - Random graph (p=1): low clustering, short path length
    - Small-world (intermediate p): high clustering AND short path length

    This model is useful for modeling social networks and neural networks.
    """
    logger.info(f"Generating Watts-Strogatz graph (n={n}, k={k}, p={p})")

    G = nx.watts_strogatz_graph(n, k, p, seed=seed)

    logger.info(
        f"Generated small-world graph: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )
    return G


def generate_powerlaw_cluster(
    n: int,
    m: int,
    p: float,
    seed: Optional[int] = None,
) -> nx.Graph:
    """Generate Holme-Kim powerlaw cluster graph.

    Creates a scale-free graph with higher clustering than Barabási-Albert,
    using triangle formation mechanism.

    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of random edges to add for each new node
    p : float
        Probability of adding a triangle after adding a random edge (0 ≤ p ≤ 1)
    seed : Optional[int]
        Random seed for reproducibility

    Returns
    -------
    nx.Graph
        Generated powerlaw cluster graph

    Examples
    --------
    >>> G = generate_powerlaw_cluster(n=1000, m=3, p=0.5, seed=42)
    >>> G.number_of_nodes()
    1000

    Notes
    -----
    The Holme-Kim model extends Barabási-Albert to include:
    - Power-law degree distribution (like BA)
    - Higher clustering coefficient (via triangle formation)

    This model better represents real networks where clustering matters,
    such as social networks and collaboration networks.
    """
    logger.info(f"Generating Holme-Kim powerlaw cluster graph (n={n}, m={m}, p={p})")

    G = nx.powerlaw_cluster_graph(n, m, p, seed=seed)

    logger.info(
        f"Generated powerlaw cluster graph: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )
    return G
