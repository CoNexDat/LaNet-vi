"""Example: Drawing a NetworkX graph with LaNet-vi, next to a spring layout.

LaNet-vi takes NetworkX graphs as they are. This example builds a synthetic Internet
topology with ``nx.random_internet_as_graph`` (2,000 autonomous systems) and draws it twice
on one matplotlib figure:

1. With NetworkX's own ``nx.spring_layout``, nodes colored by ``nx.core_number``.
2. With ``lanet_vi.draw``, which places nodes by shell index: the densest core (the
   5-shell) in the middle and lower shells further out, down to the 1-shell on the edge.

It also checks that LaNet-vi's shells are the k-cores that ``nx.core_number`` finds.

Run it with::

    uv run python examples/networkx_example.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

import lanet_vi
from lanet_vi import LaNetConfig, Network

OUTPUT = Path(__file__).parent / "outputs" / "networkx_internet_as.png"


def check_shells(graph: nx.Graph) -> int:
    """Check that LaNet-vi's k-core shells match ``nx.core_number``; return the top shell."""
    # Network numbers nodes with integers (lanet_vi.draw relabels any graph the same way)
    integer_graph = nx.convert_node_labels_to_integers(graph)
    network = Network(integer_graph)
    decomposition = network.decompose()
    if decomposition.node_indices != nx.core_number(integer_graph):
        raise AssertionError("LaNet-vi's shells differ from nx.core_number")
    return decomposition.max_index


def main(output: Path = OUTPUT) -> Path:
    """Draw a synthetic Internet AS graph with a spring layout and with LaNet-vi."""
    graph = nx.random_internet_as_graph(2000, seed=7)
    top_shell = check_shells(graph)
    print(f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, shells 1-{top_shell}")
    print("LaNet-vi's shells match nx.core_number")

    fig, (left, right) = plt.subplots(1, 2, figsize=(18, 9))

    # 1. NetworkX's own drawing: spring layout, nodes colored by core number
    core = nx.core_number(graph)
    pos = nx.spring_layout(graph, seed=42)
    nx.draw(
        graph,
        pos,
        ax=left,
        node_size=5,
        node_color=[core[node] for node in graph],
        cmap="viridis",
        edge_color="lightgray",
        width=0.3,
    )
    left.set_title("nx.spring_layout, colored by nx.core_number")

    # 2. LaNet-vi on axes of the same figure: rings by shell index (on white, to match)
    config = LaNetConfig()
    config.visualization.background = "white"
    lanet_vi.draw(graph, ax=right, config=config)
    right.set_title("lanet_vi.draw: rings by shell index")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")
    return output


if __name__ == "__main__":
    main()
