"""Input/output functions for reading network data."""

import bz2
import gzip
import io
from pathlib import Path
from typing import IO, Any

import networkx as nx
import numpy as np
import pandas as pd
import requests

from lanet_vi.logging_config import get_logger

logger = get_logger(__name__)


def _open_text(file_path: Path) -> IO[str]:
    """Open a possibly compressed (.bz2/.gz) file in text mode."""
    if file_path.suffix == ".bz2":
        return bz2.open(file_path, "rt")
    if file_path.suffix == ".gz":
        return gzip.open(file_path, "rt")
    return open(file_path)


def _integer_column(column: pd.Series, file_path: Path) -> np.ndarray:
    """Return ``column`` as an int array, rejecting non-numeric or fractional ids."""
    values = pd.to_numeric(column, errors="coerce")
    if values.isna().any() or not np.all(np.mod(values.to_numpy(), 1) == 0):
        bad = column[values.isna() | (np.mod(values, 1) != 0)].iloc[0]
        raise ValueError(f"{file_path}: node ids must be integers (found {bad!r})")
    return np.asarray(values.to_numpy(), dtype=np.int64)


def read_edge_list(
    file_path: Path | str,
    weighted: bool = False,
    directed: bool = False,
    multigraph: bool = False,
    delimiter: str | None = None,
    comment: str = "#",
) -> nx.Graph:
    """
    Read an edge list file and create a NetworkX graph.

    Lines hold ``source target [weight]`` separated by whitespace (or by
    ``delimiter`` if given). Behavior follows the C++ LaNet-vi reader: an
    unused third column is ignored, a missing weight on a weighted graph counts
    as 1.0, and self-loops are dropped (with a warning) because the
    decompositions do not accept them.

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
    delimiter : Optional[str]
        Column delimiter; ``None`` (default) accepts any run of whitespace
    comment : str
        Comment character to skip lines

    Returns
    -------
    nx.Graph
        NetworkX graph constructed from the edge list

    Raises
    ------
    ValueError
        If the file has fewer than two columns or non-integer node ids

    Examples
    --------
    >>> g = read_edge_list("network.txt", weighted=True)
    >>> g = read_edge_list("network.txt.bz2", weighted=False, directed=True)
    """
    file_path = Path(file_path)

    compression = {".bz2": "bz2", ".gz": "gzip"}.get(file_path.suffix, "none")
    logger.info(f"Reading edge list from {file_path} (compression: {compression})")

    sep = delimiter if delimiter is not None else r"\s+"
    columns = ["source", "target", "weight"]
    # Only a structurally missing field may become NaN; tokens such as "NA" or "nan"
    # must reach the numeric validation below and be rejected
    na: dict[str, Any] = {"keep_default_na": False, "na_values": []}
    try:
        # Fixed three-column schema: short rows get NaN in the missing fields and
        # extra fields are ignored, whatever the first row looks like
        with _open_text(file_path) as f:
            df = pd.read_csv(
                f, sep=sep, comment=comment, header=None, names=columns, usecols=columns, **na
            )
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=columns)
    except pd.errors.ParserError:
        # The C engine rejects usecols when no row has three fields; without usecols
        # it still pads short rows with NaN, so retry that way
        with _open_text(file_path) as f:
            df = pd.read_csv(f, sep=sep, comment=comment, header=None, names=columns, **na)

    # With default NA handling off, a structurally missing field arrives as ""
    df = df.mask(df == "")

    if df.empty:
        logger.warning(f"{file_path}: no edges found")
    missing_target = df["target"].isna()
    if missing_target.any():
        row = int(np.flatnonzero(missing_target.to_numpy())[0]) + 1
        raise ValueError(f"{file_path}: row {row} has fewer than two columns (source target)")
    has_weight_column = df["weight"].notna().any()

    source = _integer_column(df["source"], file_path)
    target = _integer_column(df["target"], file_path)

    if weighted:
        if has_weight_column:
            raw = df["weight"]
            weight_series = pd.to_numeric(raw, errors="coerce")
            bad = weight_series.isna() & raw.notna()
            if bad.any():
                first = int(np.flatnonzero(bad.to_numpy())[0])
                raise ValueError(
                    f"{file_path}: weight {raw.iloc[first]!r} on row {first + 1} is not a number"
                )
            # A missing third field (short row) counts as weight 1.0, as in the C++ reader
            weight = weight_series.fillna(1.0).to_numpy()
        else:
            logger.warning(f"{file_path}: --weighted given but no weight column; using 1.0")
            weight = np.ones(len(df))
    elif has_weight_column:
        logger.info(f"{file_path}: ignoring extra columns (graph is not weighted)")

    # Create appropriate graph type
    if directed:
        G: nx.Graph = nx.MultiDiGraph() if multigraph else nx.DiGraph()
    else:
        G = nx.MultiGraph() if multigraph else nx.Graph()

    self_loops = source == target
    n_self_loops = int(self_loops.sum())
    if n_self_loops:
        logger.warning(f"{file_path}: dropping {n_self_loops} self-loop(s)")
        # Keep nodes that only appear in a self-loop
        G.add_nodes_from(np.unique(source[self_loops]).tolist())
        source, target = source[~self_loops], target[~self_loops]
        if weighted:
            weight = weight[~self_loops]

    if weighted:
        G.add_weighted_edges_from(
            zip(source.tolist(), target.tolist(), weight.tolist(), strict=True)
        )
    else:
        G.add_edges_from(zip(source.tolist(), target.tolist(), strict=True))

    logger.info(
        f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
        f"(directed={directed}, weighted={weighted}, multigraph={multigraph})"
    )

    return G


def read_caida_snapshot(
    url: str,
    timeout: int = 30,
) -> tuple[nx.Graph, pd.DataFrame]:
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
        raise ValueError(f"Failed to decompress data from {url}: {e}") from e

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
        raise ValueError(f"Failed to parse CSV data from {url}: {e}") from e

    # Create graph (convert to int to avoid float64 node IDs)
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(
            int(row["provider"]), int(row["customer"]), relationship=int(row["relationship_type"])
        )

    logger.info(f"Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G, df


def read_node_names(
    file_path: Path | str,
    delimiter: str | None = None,
    comment: str = "#",
) -> dict[int, str]:
    """
    Read node names from a file.

    Each line is ``node_id name``; the name is everything after the first
    separator, so it may contain spaces. Surrounding quotes are removed, as in
    the C++ reader.

    Parameters
    ----------
    file_path : Union[Path, str]
        Path to file with node names (format: node_id name)
    delimiter : Optional[str]
        Separator between the id and the name; ``None`` (default) means any
        run of whitespace
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
    file_path = Path(file_path)
    logger.info(f"Reading node names from {file_path}")

    names_dict: dict[int, str] = {}
    with _open_text(file_path) as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.split(comment, 1)[0].strip() if comment else raw.strip()
            if not line:
                continue
            parts = line.split(delimiter, 1)
            try:
                node_id = int(parts[0])
            except ValueError as exc:
                raise ValueError(f"{file_path}:{lineno}: node id must be an integer") from exc
            name = parts[1].strip() if len(parts) > 1 else ""
            if len(name) >= 2 and name[0] == name[-1] and name[0] in "\"'":
                name = name[1:-1]
            names_dict[node_id] = name

    logger.info(f"Loaded {len(names_dict)} node names")

    return names_dict


def read_node_colors(
    file_path: Path | str,
    delimiter: str | None = None,
    comment: str = "#",
) -> dict[int, tuple[float, float, float]]:
    """
    Read node colors from a file.

    Parameters
    ----------
    file_path : Union[Path, str]
        Path to file with node colors (format: node_id r g b)
        RGB values should be in range [0.0, 1.0]
    delimiter : Optional[str]
        Column delimiter; ``None`` (default) accepts any run of whitespace
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
        sep=delimiter if delimiter is not None else r"\s+",
        comment=comment,
        names=["node_id", "r", "g", "b"],
        dtype={"node_id": int, "r": float, "g": float, "b": float},
    )

    # Validate RGB values
    if not ((df[["r", "g", "b"]] >= 0.0) & (df[["r", "g", "b"]] <= 1.0)).all().all():
        logger.error("RGB values must be in range [0.0, 1.0]")
        raise ValueError("RGB values must be in range [0.0, 1.0]")

    colors = {
        int(node_id): (float(r), float(g), float(b))
        for node_id, r, g, b in df[["node_id", "r", "g", "b"]].itertuples(index=False)
    }

    logger.info(f"Loaded {len(colors)} node colors")

    return colors
