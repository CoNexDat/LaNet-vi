"""Output functions for writing decomposition results and visualizations."""

import json
from pathlib import Path
from typing import Dict, Optional, Union

import networkx as nx
import pandas as pd

from lanet_vi.logging_config import get_logger
from lanet_vi.models.graph import DecompositionResult

logger = get_logger(__name__)


def write_decomposition_csv(
    result: DecompositionResult,
    output_path: Union[Path, str],
) -> None:
    """
    Export decomposition results to CSV.

    Parameters
    ----------
    result : DecompositionResult
        Decomposition result to export
    output_path : Union[Path, str]
        Output file path

    Examples
    --------
    >>> write_decomposition_csv(decomp_result, "cores.csv")
    """
    output_path = Path(output_path)

    logger.info(f"Exporting {result.decomp_type} decomposition to CSV: {output_path}")

    df = pd.DataFrame(
        [(node_id, index) for node_id, index in result.node_indices.items()],
        columns=["node_id", f"{result.decomp_type}_index"],
    )

    df.to_csv(output_path, index=False)

    logger.info(f"Wrote {len(df)} nodes to {output_path}")


def write_decomposition_json(
    result: DecompositionResult,
    output_path: Union[Path, str],
    include_components: bool = True,
    include_metadata: bool = True,
) -> None:
    """
    Export decomposition results to JSON with enhanced details.

    Parameters
    ----------
    result : DecompositionResult
        Decomposition result to export
    output_path : Union[Path, str]
        Output file path
    include_components : bool
        Include component information (default: True)
    include_metadata : bool
        Include metadata (default: True)

    Examples
    --------
    >>> write_decomposition_json(decomp_result, "cores.json")
    >>> write_decomposition_json(decomp_result, "cores_full.json", include_components=True)
    """
    output_path = Path(output_path)

    logger.info(f"Exporting {result.decomp_type} decomposition to JSON: {output_path}")

    data = {
        "decomposition_type": result.decomp_type,
        "max_index": result.max_index,
        "min_index": result.min_index,
        "num_nodes": len(result.node_indices),
        "num_components": len(result.components),
        "node_indices": {str(k): v for k, v in result.node_indices.items()},
    }

    if result.p_function:
        data["p_function"] = result.p_function

    # Include component details
    if include_components and result.components:
        data["components"] = [
            {
                "id": comp.id,
                "index": comp.index,
                "size": comp.size,
                "nodes": comp.nodes,
            }
            for comp in result.components
        ]

        # Add component statistics
        component_sizes = [comp.size for comp in result.components]
        data["component_statistics"] = {
            "total_components": len(result.components),
            "largest_component_size": max(component_sizes) if component_sizes else 0,
            "smallest_component_size": min(component_sizes) if component_sizes else 0,
            "mean_component_size": sum(component_sizes) / len(component_sizes) if component_sizes else 0,
        }

    # Include metadata (d-cores pairs, community info, etc.)
    if include_metadata and result.metadata:
        # Handle special metadata types
        processed_metadata = {}
        for key, value in result.metadata.items():
            if key == "d_cores":
                # Convert (k_in, k_out) tuples to lists for JSON serialization
                processed_metadata[key] = {
                    str(node): list(pair) if isinstance(pair, tuple) else pair
                    for node, pair in value.items()
                }
            else:
                processed_metadata[key] = value

        data["metadata"] = processed_metadata

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Wrote decomposition with {len(result.node_indices)} nodes to {output_path}")


def write_node_attributes(
    node_data: Dict[int, Dict],
    output_path: Union[Path, str],
) -> None:
    """
    Export node attributes to CSV.

    Parameters
    ----------
    node_data : Dict[int, Dict]
        Mapping from node ID to attribute dictionary
    output_path : Union[Path, str]
        Output file path

    Examples
    --------
    >>> attrs = {1: {"shell": 3, "x": 0.5, "y": 0.3}, 2: {"shell": 2, "x": 0.2, "y": 0.1}}
    >>> write_node_attributes(attrs, "nodes.csv")
    """
    output_path = Path(output_path)

    logger.info(f"Exporting node attributes to CSV: {output_path}")

    df = pd.DataFrame.from_dict(node_data, orient="index")
    df.index.name = "node_id"
    df.to_csv(output_path)

    logger.info(f"Wrote attributes for {len(df)} nodes to {output_path}")


def write_graph_json(
    graph: nx.Graph,
    output_path: Union[Path, str],
    include_node_attrs: bool = True,
    include_edge_attrs: bool = True,
) -> None:
    """
    Export NetworkX graph to JSON using node-link format.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph to export
    output_path : Union[Path, str]
        Output file path
    include_node_attrs : bool
        Include node attributes (default: True)
    include_edge_attrs : bool
        Include edge attributes (default: True)

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> write_graph_json(G, "karate.json")

    Notes
    -----
    Uses NetworkX's node-link JSON format, which is compatible with D3.js
    and other visualization libraries.
    """
    output_path = Path(output_path)

    logger.info(
        f"Exporting graph to JSON: {output_path} "
        f"({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)"
    )

    # Convert to node-link format
    data = nx.node_link_data(graph)

    # Optionally strip attributes
    if not include_node_attrs:
        for node in data["nodes"]:
            node.clear()
            node["id"] = node.get("id", 0)

    if not include_edge_attrs:
        for edge in data["links"]:
            edge_keys = list(edge.keys())
            for key in edge_keys:
                if key not in ("source", "target"):
                    del edge[key]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Wrote graph to {output_path}")


def write_community_json(
    community_result,  # CommunityResult type
    output_path: Union[Path, str],
) -> None:
    """
    Export community detection results to JSON.

    Parameters
    ----------
    community_result : CommunityResult
        Community detection result
    output_path : Union[Path, str]
        Output file path

    Examples
    --------
    >>> from lanet_vi.community import detect_communities_louvain
    >>> communities = detect_communities_louvain(G)
    >>> write_community_json(communities, "communities.json")
    """
    output_path = Path(output_path)

    logger.info(f"Exporting community detection results to JSON: {output_path}")

    data = {
        "algorithm": community_result.algorithm,
        "num_communities": community_result.num_communities,
        "modularity": community_result.modularity,
        "communities": [
            {
                "id": comm.id,
                "size": comm.size,
                "nodes": comm.nodes,
            }
            for comm in community_result.communities
        ],
        "node_to_community": {
            str(k): v for k, v in community_result.node_to_community.items()
        },
    }

    # Add community size statistics
    sizes = [comm.size for comm in community_result.communities]
    data["statistics"] = {
        "largest_community": max(sizes) if sizes else 0,
        "smallest_community": min(sizes) if sizes else 0,
        "mean_community_size": sum(sizes) / len(sizes) if sizes else 0,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(
        f"Wrote {community_result.num_communities} communities to {output_path}"
    )


def write_edge_list(
    graph: nx.Graph,
    output_path: Union[Path, str],
    include_weights: bool = True,
    delimiter: str = "\t",
) -> None:
    """
    Write graph to edge list file using pandas.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph
    output_path : Union[Path, str]
        Output file path
    include_weights : bool
        Include edge weights if available (default: True)
    delimiter : str
        Column delimiter (default: tab)

    Examples
    --------
    >>> write_edge_list(G, "network.txt", delimiter=" ")
    """
    output_path = Path(output_path)

    logger.info(f"Writing edge list to {output_path}")

    edges = []
    for u, v, data in graph.edges(data=True):
        if include_weights and "weight" in data:
            edges.append((u, v, data["weight"]))
        else:
            edges.append((u, v))

    # Create DataFrame
    if include_weights and edges and len(edges[0]) == 3:
        df = pd.DataFrame(edges, columns=["source", "target", "weight"])
    else:
        df = pd.DataFrame(edges, columns=["source", "target"])

    # Write to file
    df.to_csv(output_path, sep=delimiter, index=False, header=False)

    logger.info(f"Wrote {len(edges)} edges to {output_path}")
