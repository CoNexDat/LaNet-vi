"""Input/output functions for reading network data."""

import bz2
import gzip
import io
from pathlib import Path
from typing import Dict, Tuple, Union

import networkx as nx
import pandas as pd
import requests

from lanet_vi.logging_config import get_logger

logger = get_logger(__name__)


def read_edge_list(
    file_path: Union[Path, str],
    weighted: bool = False,
    directed: bool = False,
    multigraph: bool = False,
    delimiter: str = " ",
    comment: str = "#",
) -> nx.Graph:
    """
    Read an edge list file and create a NetworkX graph.

    Parameters
    ----------
    file_path : Union[Path, str]
        Path to the edge list file (supports .txt, .gz, .bz2)
    weighted : bool
        Whether edges have weights (third column)
    directed : bool
        Whether to create a directed graph
    multigraph : bool
        Whether to allow parallel edges
    delimiter : str
        Column delimiter in the file
    comment : str
        Comment character to skip lines

    Returns
    -------
    nx.Graph
        NetworkX graph constructed from the edge list

    Examples
    --------
    >>> g = read_edge_list("network.txt", weighted=True)
    >>> g = read_edge_list("network.txt.bz2", weighted=False, directed=True)
    """
    file_path = Path(file_path)

    # Determine compression
    if file_path.suffix == ".bz2":
        open_func = bz2.open
        compression = "bz2"
    elif file_path.suffix == ".gz":
        open_func = gzip.open
        compression = "gzip"
    else:
        open_func = open
        compression = "none"

    logger.info(f"Reading edge list from {file_path} (compression: {compression})")

    # Read edge list with pandas
    with open_func(file_path, "rt") as f:
        if weighted:
            df = pd.read_csv(
                f,
                sep=delimiter,
                comment=comment,
                names=["source", "target", "weight"],
                dtype={"source": int, "target": int, "weight": float},
            )
        else:
            df = pd.read_csv(
                f,
                sep=delimiter,
                comment=comment,
                names=["source", "target"],
                dtype={"source": int, "target": int},
            )
            df["weight"] = 1.0

    # Create appropriate graph type
    if directed:
        if multigraph:
            G = nx.MultiDiGraph()
        else:
            G = nx.DiGraph()
    else:
        if multigraph:
            G = nx.MultiGraph()
        else:
            G = nx.Graph()

    # Add edges (use values for speed, avoid iterrows)
    if weighted or multigraph:
        edges_with_weights = [(int(row[0]), int(row[1]), row[2])
                              for row in df.values]
        G.add_weighted_edges_from(edges_with_weights)
    else:
        edges = [(int(row[0]), int(row[1])) for row in df.values]
        G.add_edges_from(edges)

    logger.info(
        f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
        f"(directed={directed}, weighted={weighted}, multigraph={multigraph})"
    )

    return G


def read_caida_snapshot(
    url: str,
    timeout: int = 30,
) -> Tuple[nx.Graph, pd.DataFrame]:
    """
    Fetch and parse CAIDA AS-Relationships data.

    This function downloads a CAIDA AS-relationships snapshot in bz2 format,
    decompresses it, and creates both a NetworkX graph and a pandas DataFrame.

    Parameters
    ----------
    url : str
        URL to the CAIDA .as-rel.txt.bz2 file
    timeout : int
        Request timeout in seconds

    Returns
    -------
    graph : nx.Graph
        NetworkX graph with AS relationships
    dataframe : pd.DataFrame
        DataFrame with columns: provider, customer, relationship_type

    Raises
    ------
    requests.RequestException
        If the download fails
    ValueError
        If decompression or parsing fails

    Examples
    --------
    >>> url = "https://publicdata.caida.org/.../20251001.as-rel.txt.bz2"
    >>> graph, df = read_caida_snapshot(url)
    """
    logger.info(f"Downloading CAIDA snapshot from {url}")

    # Download data
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    logger.debug(f"Downloaded {len(response.content)} bytes")

    # Decompress
    try:
        decompressed_data = bz2.decompress(response.content)
        logger.debug(f"Decompressed to {len(decompressed_data)} bytes")
    except Exception as e:
        logger.error(f"Failed to decompress data from {url}: {e}")
        raise ValueError(f"Failed to decompress data from {url}: {e}")

    # Parse CSV
    try:
        data_io = io.StringIO(decompressed_data.decode("utf-8"))
        df = pd.read_csv(
            data_io,
            sep="|",
            comment="#",
            names=["provider", "customer", "relationship_type"],
            dtype={"provider": int, "customer": int, "relationship_type": int},
        )
        logger.info(f"Parsed {len(df)} AS relationships")
    except Exception as e:
        logger.error(f"Failed to parse CSV data from {url}: {e}")
        raise ValueError(f"Failed to parse CSV data from {url}: {e}")

    # Create graph (convert to int to avoid float64 node IDs)
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(
            int(row["provider"]),
            int(row["customer"]),
            relationship=int(row["relationship_type"])
        )

    logger.info(f"Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G, df


def read_node_names(
    file_path: Union[Path, str],
    delimiter: str = " ",
    comment: str = "#",
) -> Dict[int, str]:
    """
    Read node names from a file.

    Parameters
    ----------
    file_path : Union[Path, str]
        Path to file with node names (format: node_id name)
    delimiter : str
        Column delimiter
    comment : str
        Comment character

    Returns
    -------
    Dict[int, str]
        Mapping from node ID to node name

    Examples
    --------
    >>> names = read_node_names("nodes.txt")
    >>> names[42]
    'node_name_42'
    """
    logger.info(f"Reading node names from {file_path}")

    df = pd.read_csv(
        file_path,
        sep=delimiter,
        comment=comment,
        names=["node_id", "name"],
        dtype={"node_id": int, "name": str},
    )

    names_dict = dict(zip(df["node_id"], df["name"]))
    logger.info(f"Loaded {len(names_dict)} node names")

    return names_dict


def read_node_colors(
    file_path: Union[Path, str],
    delimiter: str = " ",
    comment: str = "#",
) -> Dict[int, Tuple[float, float, float]]:
    """
    Read node colors from a file.

    Parameters
    ----------
    file_path : Union[Path, str]
        Path to file with node colors (format: node_id r g b)
        RGB values should be in range [0.0, 1.0]
    delimiter : str
        Column delimiter
    comment : str
        Comment character

    Returns
    -------
    Dict[int, Tuple[float, float, float]]
        Mapping from node ID to (r, g, b) tuple

    Examples
    --------
    >>> colors = read_node_colors("colors.txt")
    >>> colors[42]
    (1.0, 0.0, 0.0)  # Red
    """
    logger.info(f"Reading node colors from {file_path}")

    df = pd.read_csv(
        file_path,
        sep=delimiter,
        comment=comment,
        names=["node_id", "r", "g", "b"],
        dtype={"node_id": int, "r": float, "g": float, "b": float},
    )

    # Validate RGB values
    if not ((df[["r", "g", "b"]] >= 0.0) & (df[["r", "g", "b"]] <= 1.0)).all().all():
        logger.error("RGB values must be in range [0.0, 1.0]")
        raise ValueError("RGB values must be in range [0.0, 1.0]")

    colors = {}
    for _, row in df.iterrows():
        colors[row["node_id"]] = (row["r"], row["g"], row["b"])

    logger.info(f"Loaded {len(colors)} node colors")

    return colors
