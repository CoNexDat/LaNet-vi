"""Use LaNet-vi from NetworkX code, like one of NetworkX's own drawing functions.

:func:`lanetvi_layout` returns node positions the way ``nx.spring_layout`` does, so it can
be passed to ``nx.draw`` and the rest of NetworkX's drawing API; :func:`draw` draws the
full LaNet-vi picture (rings, colors, legends) on matplotlib axes. Both take any NetworkX
graph as it is: nodes may be strings, tuples or any other hashable, and edge weights and
direction are detected as :class:`~lanet_vi.core.network.Network` does.

Examples
--------
>>> import networkx as nx
>>> import lanet_vi
>>> G = nx.les_miserables_graph()
>>> fig = lanet_vi.draw(G)  # the LaNet-vi picture
>>> pos = lanet_vi.lanetvi_layout(G)  # or only the positions
>>> nx.draw(G, pos, node_size=20)
"""

from collections.abc import Hashable

import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from lanet_vi.core.network import Network
from lanet_vi.models.config import LaNetConfig

__all__ = ["draw", "lanetvi_layout"]


def _is_integer_label(node: Hashable) -> bool:
    """Tell whether LaNet-vi takes the label as it is: a Python ``int``, not a ``bool``."""
    return isinstance(node, int) and not isinstance(node, bool)


def _integer_graph(graph: nx.Graph) -> tuple[nx.Graph, dict[int, Hashable] | None]:
    """Relabel the graph with integers and return the original label of each integer.

    LaNet-vi numbers nodes with integers. A graph whose nodes already are integers is
    returned as it is with no mapping; any other is relabeled ``0..n-1`` in node order.
    """
    if all(_is_integer_label(node) for node in graph.nodes):
        return graph, None
    labels = dict(enumerate(graph.nodes))
    relabeled = nx.relabel_nodes(graph, {node: i for i, node in labels.items()}, copy=True)
    return relabeled, labels


def _network(graph: nx.Graph, config: LaNetConfig | None) -> tuple[Network, dict[int, Hashable]]:
    """Build and decompose a :class:`Network` of ``graph``; map its nodes to their labels."""
    if graph.number_of_nodes() == 0:
        raise ValueError("The graph has no nodes")
    integer_graph, labels = _integer_graph(graph)
    network = Network(integer_graph, config)
    if labels is not None:
        # Labels (config.visualization.show_node_labels) show the original names
        network.node_names = {i: str(label) for i, label in labels.items()}
    network.decompose()
    return network, labels if labels is not None else {node: node for node in graph.nodes}


def lanetvi_layout(
    G: nx.Graph,
    config: LaNetConfig | None = None,
    scale: float | None = 1.0,
    center: tuple[float, float] | None = None,
) -> dict[Hashable, np.ndarray]:
    """
    Position the nodes in LaNet-vi's concentric rings.

    Runs the decomposition chosen in ``config`` (k-cores by default) and returns the
    layout as a NetworkX layout does, a dict from node to an ``(x, y)`` array, so it can
    be used wherever ``nx.spring_layout`` is.

    Parameters
    ----------
    G : nx.Graph
        Any NetworkX graph (``Graph``, ``DiGraph``, ``MultiGraph``); node labels can be
        any hashable
    config : Optional[LaNetConfig]
        LaNet-vi settings (defaults when None)
    scale : Optional[float]
        Center the positions on their mean and scale them uniformly so the largest
        absolute coordinate is ``scale``, as NetworkX layouts do (``nx.rescale_layout``);
        None keeps LaNet-vi's own units
    center : Optional[Tuple[float, float]]
        Center of the layout; the origin when None

    Returns
    -------
    Dict[Hashable, np.ndarray]
        Position of every node that is drawn. With ``config.decomposition.from_layer``
        set, the nodes below that layer are not drawn and have no position.

    Raises
    ------
    ValueError
        If the graph has no nodes

    Examples
    --------
    >>> G = nx.karate_club_graph()
    >>> pos = lanetvi_layout(G)
    >>> nx.draw(G, pos)
    """
    network, labels = _network(G, config)
    layout = network.compute_layout()
    pos = {labels[node]: np.array(xy, dtype=float) for node, xy in layout.node_positions.items()}
    if scale is not None and pos:
        pos = nx.rescale_layout_dict(pos, scale=scale)
    if center is not None:
        offset = np.asarray(center, dtype=float)
        pos = {node: xy + offset for node, xy in pos.items()}
    return pos


def draw(
    G: nx.Graph,
    *,
    ax: Axes | None = None,
    config: LaNetConfig | None = None,
) -> Figure:
    """
    Draw a NetworkX graph with LaNet-vi.

    The picture is the one ``lanet-vi visualize`` saves: nodes in concentric rings by
    k-core (or the decomposition chosen in ``config``), colored by index, with the
    legends. It is drawn on matplotlib axes and nothing is saved.

    Unlike ``nx.draw``, it computes its own positions (there is no ``pos`` argument) and
    returns the figure. A new figure is registered with pyplot like any ``plt.figure()``:
    close it with ``plt.close(fig)`` when drawing many graphs in a loop.

    Parameters
    ----------
    G : nx.Graph
        Any NetworkX graph (``Graph``, ``DiGraph``, ``MultiGraph``); node labels can be
        any hashable, and node labels in the picture show them as text
    ax : Optional[Axes]
        Axes to draw on (keyword only); a new figure of
        ``config.visualization.width x height`` pixels when None. See
        :func:`~lanet_vi.visualization.matplotlib_renderer.draw_network` for what is
        changed on given axes.
    config : Optional[LaNetConfig]
        LaNet-vi settings (keyword only; defaults when None)

    Returns
    -------
    Figure
        The figure of the axes the graph was drawn on (the root figure when the axes are
        in a subfigure)

    Raises
    ------
    ValueError
        If the graph has no nodes
    TypeError
        If ``ax`` is not matplotlib axes

    Examples
    --------
    >>> G = nx.les_miserables_graph()
    >>> fig = draw(G)
    >>> fig.savefig("les_miserables.png")
    """
    if ax is not None and not isinstance(ax, Axes):
        raise TypeError(f"ax must be matplotlib Axes, not {type(ax).__name__}")
    network, _ = _network(G, config)
    return network.draw(ax=ax)
