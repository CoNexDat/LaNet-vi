"""Connected pieces of each level of a decomposition, in one pass over the edges."""

from __future__ import annotations

import networkx as nx

from lanet_vi.models.graph import Component
from lanet_vi.union_find import UnionFind


def components_by_index(
    graph: nx.Graph, node_indices: dict[int, int], *, dense: bool = False
) -> list[Component]:
    """Split every level of a decomposition into its connected pieces.

    A piece of index ``k`` is a (weakly) connected component of the subgraph induced by
    the nodes of index ``k``. One union-find pass over the edges replaces one induced
    subgraph per level, which copied the top of a deep hierarchy once per level.

    Parameters
    ----------
    graph : nx.Graph
        The network; direction and parallel edges do not matter for connectivity
    node_indices : Dict[int, int]
        Index of every node of ``graph`` (nodes missing from it are skipped)
    dense : bool
        Store the index as ``dense_index`` (k-dense) instead of ``shell_index``

    Returns
    -------
    List[Component]
        Highest index first; within an index, pieces in the order of their first node in
        ``graph.nodes()``, each listing its nodes in that order; ids ``0, 1, ...``
    """
    nodes = [v for v in graph.nodes() if v in node_indices]
    position = {v: i for i, v in enumerate(nodes)}
    sets = UnionFind(len(nodes))
    for u, v in graph.edges():
        if u in position and v in position and node_indices[u] == node_indices[v]:
            sets.union(position[u], position[v])

    # Pieces keyed by index, then by representative, both in first-node order
    pieces: dict[int, dict[int, list[int]]] = {}
    for v in nodes:
        pieces.setdefault(node_indices[v], {}).setdefault(sets.find(position[v]), []).append(v)

    components: list[Component] = []
    for index in sorted(pieces, reverse=True):
        for members in pieces[index].values():
            key = "dense_index" if dense else "shell_index"
            components.append(
                Component(
                    component_id=len(components),
                    nodes=members,
                    size=len(members),
                    **{key: index},
                )
            )
    return components
