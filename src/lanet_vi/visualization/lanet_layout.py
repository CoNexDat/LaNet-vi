"""The LaNet-vi placement: nested components, neighbour-based rho, clique sectors.

Port of ``kcores_component.cpp`` / ``graph_kcores_components.cpp`` from the C++ LaNet-vi
(``-coordDistributionAlgorithm classic``, the default), following the formulas of
Alvarez-Hamelin, Dall'Asta, Barrat & Vespignani, *k-core decomposition: a tool for the
visualization of large scale networks* (NIPS 2005):

1. The graph is split recursively into nested **components**: the connected components of
   the subgraph with index ``> k`` inside a component of index ``k``. A component of index
   ``k`` owns the **clusters** of its nodes with index exactly ``k`` (connected inside the
   shell) and its children (index ``k + 1``).
2. Each component gets a centre, a radius (``ratio``) and a scale (``u``): the top core of
   a branch has a radius proportional to the root of the sum of its squared log-degrees,
   every enclosing shell adds one unit (part 1); children are placed inside their parent
   at ``rho = 1 - size / siblings``, ``phi = 2 pi (previous / siblings)^2`` and scaled by
   ``sqrt(size / siblings) / delta`` (formulas (3)-(5), part 2).
3. A node of index ``k`` sits at ``rho = ratio (1 - eps) + eps ratio * average``, where
   ``average`` measures how deep its higher-index neighbours are (formula (1)), and at the
   circular average of the angles of those neighbours (already placed). Top cores are
   split into cliques laid along U-shaped paths in angular sectors (formula (2)).

K-dense and d-core results go through the same classic placement with their own *edge
index* for the component tree (an edge belongs to the inner component when its index is
above the component's: for k-cores the minimum of its endpoints' indices, for k-dense the
edge's own dense index, as ``kdenses_component.cpp`` walks it). The rest of
``kdenses_component.cpp`` (``>=`` neighbour selection, the ``tau`` factor, sibling circle
packing through ``distribute_components``, ``ratioConstant`` node radii) belongs to the
C++ "modern" ``pow``/``log`` mode, which is not ported yet.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

TWO_PI = 2.0 * math.pi


@dataclass
class LayoutComponent:
    """A component of the nested decomposition with its placement (``KCores_Component``)."""

    index: int
    parent: LayoutComponent | None = None
    clusters: list[list[int]] = field(default_factory=list)
    children: list[LayoutComponent] = field(default_factory=list)
    size: int = 0
    shell_cardinal: int = 0
    rho: float = -1.0
    phi: float = 0.0
    x: float = 0.0
    y: float = 0.0
    u: float = 1.0
    ratio: float = 1.0
    central_core_k: int = 0
    central_core_ratio: float = 1.0

    def walk(self) -> list[LayoutComponent]:
        """Return this component and all its descendants, parents first."""
        out: list[LayoutComponent] = []
        stack = [self]
        while stack:
            comp = stack.pop()
            out.append(comp)
            stack.extend(reversed(comp.children))
        return out


@dataclass
class LayoutParameters:
    """The C++ parameters that shape the picture."""

    epsilon: float = 0.18
    delta: float = 1.3
    gamma: float = 1.5
    u: float = 1.0
    no_cliques: bool = False
    weighted: bool = False


@dataclass
class LanetLayout:
    """Result of :func:`compute_lanet_layout`."""

    positions: dict[int, tuple[float, float]]
    root: LayoutComponent
    #: Half-size of the square the C++ camera framed: ``gamma * u * R`` with ``R`` the
    #: radius of the outermost non-empty component.
    frame: float


def circular_average(a: float, weight_a: float, b: float, weight_b: float) -> float:
    """Weighted average of two angles along the shorter arc (``Circular_average``)."""
    a %= TWO_PI
    b %= TWO_PI
    if a > b:
        aperture, first, weight_second = a - b, b, weight_a
    else:
        aperture, first, weight_second = b - a, a, weight_b
    if aperture > math.pi:
        aperture = TWO_PI - aperture
        if first == a:
            first, weight_second = b, weight_a
        else:
            first, weight_second = a, weight_b
    return first + (aperture * weight_second) / (weight_a + weight_b)


def place_in_circular_sector(
    ratio: float, alfa: float, n: int, total: int, random: float, unique: bool
) -> tuple[float, float]:
    """Position ``n`` of ``total`` along a U-shaped path in a sector (``placeInCircularSector``).

    The path goes out along the radius, along the arc of aperture ``alfa`` and back in;
    with ``unique`` (the clique is the whole core) only the arc is used.
    """
    pos = n / total + random
    if pos > 1.0:
        pos -= 1.0
    divs = ratio + pos * ratio * alfa if unique else pos * (2.0 * ratio + ratio * alfa)
    if divs < ratio:
        return divs, 0.0
    if divs < ratio + ratio * alfa:
        return ratio, (divs - ratio) / ratio
    return ratio - (divs - ratio - ratio * alfa), alfa


def _post_order(root: LayoutComponent) -> list[LayoutComponent]:
    """Components with every child before its parent, children in their own order."""
    out: list[LayoutComponent] = []
    stack: list[tuple[LayoutComponent, bool]] = [(root, False)]
    while stack:
        comp, expanded = stack.pop()
        if expanded:
            out.append(comp)
            continue
        stack.append((comp, True))
        stack.extend((child, False) for child in reversed(comp.children))
    return out


def _random_order(items: Sequence[int], rng: np.random.Generator) -> list[int]:
    """Shuffle as the C++ did: every element goes to the front or the back with p = 1/2."""
    out: deque[int] = deque()
    for item in items:
        if rng.random() > 0.5:
            out.append(item)
        else:
            out.appendleft(item)
    return list(out)


def build_component_tree(
    graph: nx.Graph,
    node_index: dict[int, int],
    edge_index: Callable[[int, int], int],
    rng: np.random.Generator,
) -> LayoutComponent:
    """Nested components and clusters of a decomposition (``computeComponents``).

    Parameters
    ----------
    graph : nx.Graph
        The network
    node_index : Dict[int, int]
        Shell / dense index of every node
    edge_index : Callable[[int, int], int]
        Index of an edge: the component of index ``k`` is connected through edges of
        index ``> k`` and its clusters through edges of index ``== k``
    rng : numpy.random.Generator
        Source of the random cluster order

    Returns
    -------
    LayoutComponent
        The root (index 0, all nodes)
    """
    root = LayoutComponent(index=0)
    pending: list[tuple[LayoutComponent, list[int]]] = [(root, list(graph.nodes()))]
    while pending:
        comp, vertices = pending.pop()
        comp.size = len(vertices)
        seen: set[int] = set()
        shell_nodes: list[int] = []
        for v in vertices:
            if v in seen:
                continue
            if node_index[v] > comp.index:
                # A connected piece of the inner core becomes a child component
                child = LayoutComponent(index=comp.index + 1, parent=comp)
                members = [v]
                seen.add(v)
                queue = deque([v])
                while queue:
                    current = queue.popleft()
                    for w in graph.neighbors(current):
                        if w not in seen and edge_index(current, w) > comp.index:
                            seen.add(w)
                            members.append(w)
                            queue.append(w)
                comp.children.append(child)
                pending.append((child, members))
            else:
                shell_nodes.append(v)

        comp.shell_cardinal = len(shell_nodes)
        clustered: set[int] = set()
        clusters: list[list[int]] = []
        for v in shell_nodes:
            if v in clustered:
                continue
            cluster = [v]
            clustered.add(v)
            stack = [v]
            while stack:
                current = stack.pop()
                for w in graph.neighbors(current):
                    if (
                        w not in clustered
                        and node_index[w] == comp.index
                        and edge_index(current, w) == comp.index
                    ):
                        clustered.add(w)
                        cluster.append(w)
                        stack.append(w)
            clusters.append(cluster)
        comp.clusters = [clusters[i] for i in _random_order(range(len(clusters)), rng)]
    return root


def _greedy_cliques(cluster: list[int], graph: nx.Graph) -> list[list[int]]:
    """Partition a top-core cluster into cliques (``Clique::buildCliques``).

    Each node, in cluster order, seeds a clique that greedily absorbs its not-yet-taken
    neighbours, in adjacency order, when they are adjacent to every member; cliques are
    sorted by node id. The C++ meant to rank nodes by the connections among their
    top-core neighbours, but it looked the core numbers up in the empty map of a fresh
    ``Network``, so every rank was 0, the sort was a no-op and the cluster / adjacency
    orders decided; that is what is reproduced here. Linear in the edges inside the
    cluster.
    """
    members = set(cluster)
    inside = {v: set(graph[v]) & members for v in cluster}
    alive = set(cluster)
    cliques: list[list[int]] = []
    for v in cluster:
        if v not in alive:
            continue
        clique = [v]
        for w in graph[v]:
            if w in alive and w != v and all(w in inside[c] for c in clique):
                clique.append(w)
        alive.difference_update(clique)
        cliques.append(sorted(clique))
    return cliques


class _Placer:
    """Runs the two passes of ``findCoordinatesClassic`` over a component tree."""

    def __init__(
        self,
        graph: nx.Graph,
        node_index: dict[int, int],
        params: LayoutParameters,
        rng: np.random.Generator,
    ) -> None:
        self.graph = graph
        self.node_index = node_index
        self.params = params
        self.rng = rng
        self.max_index = max(node_index.values()) if node_index else 0
        self.degree = dict(graph.degree())
        # The C++ divides by log(max strength); guard the degenerate log(1) = 0
        strength = self.degree
        if params.weighted:
            strength = {
                v: sum(float(d.get("weight", 1.0)) for d in graph[v].values()) for v in graph
            }
        top = max(strength.values(), default=1.0)
        self.log_max_strength = max(math.log(max(top, 1.0)), 1e-9)
        # ``degree_squares`` is a file-scope accumulator in the C++: it keeps growing over
        # every top core visited, so later top cores get larger radii. Kept as is.
        self.degree_squares = 0.0
        self.positions: dict[int, tuple[float, float]] = {}

    # -- part 1: radii ---------------------------------------------------------------
    def radii(self, root: LayoutComponent) -> None:
        """Post-order over the tree: the radius of every component from its deepest core.

        Iterative (children in order, then the parent) so the depth of the core hierarchy
        is not bounded by Python's recursion limit.
        """
        for comp in _post_order(root):
            comp.central_core_k = comp.index
            for child in comp.children:
                if child.central_core_k > comp.central_core_k:
                    comp.central_core_k = child.central_core_k
                    comp.central_core_ratio = child.central_core_ratio
            if not comp.children:
                for cluster in comp.clusters:
                    for v in cluster:
                        self.degree_squares += math.log(1 + self.degree[v]) ** 2
                comp.central_core_k = comp.index
                comp.central_core_ratio = (
                    2
                    * math.sqrt(40 * 40 * 0.01 * 0.01 * self.degree_squares)
                    / self.log_max_strength
                )
                comp.ratio = comp.central_core_ratio
            comp.ratio = comp.central_core_ratio + (comp.central_core_k - comp.index)

    # -- part 2: centres and node positions ------------------------------------------
    def place(self, root: LayoutComponent) -> None:
        """Centre and scale every component, then its nodes once its children are placed.

        Same order as the C++ recursion: a component's centre first, then its children
        (depth first), then its own clusters. Iterative for the same reason as ``radii``.
        """
        stack: list[tuple[LayoutComponent, bool]] = [(root, False)]
        while stack:
            comp, children_done = stack.pop()
            if children_done:
                self.place_own_nodes(comp)
                continue
            self.place_centre(comp)
            stack.append((comp, True))
            stack.extend((child, False) for child in reversed(comp.children))

    def place_centre(self, comp: LayoutComponent) -> None:
        """Formulas (4), (3) and (5): a child's rho, phi, centre and scale in its parent."""
        parent = comp.parent
        if parent is None:
            return
        siblings = sum(c.size for c in parent.children)
        previous = 0
        for c in parent.children:
            if c is comp:
                break
            previous += c.size
        previous += comp.size // 2
        comp.rho = 1.0 - comp.size / siblings
        comp.phi = TWO_PI * (previous / siblings) ** 2
        comp.x = parent.x + parent.ratio * parent.u * comp.rho * math.cos(comp.phi)
        comp.y = parent.y + parent.ratio * parent.u * comp.rho * math.sin(comp.phi)
        if len(parent.children) != 1:
            comp.u = math.sqrt(comp.size / siblings) * parent.u / self.params.delta
        else:
            comp.u = parent.u

    def place_own_nodes(self, comp: LayoutComponent) -> None:
        """Place the clusters of a component: cliques for a top core, formula (1) otherwise."""
        partial = 0
        for cluster in comp.clusters:
            if cluster:
                if not comp.children and not self.params.no_cliques:
                    self.place_cliques(comp, cluster)
                else:
                    self.place_cluster(comp, cluster, partial)
            partial += len(cluster)

    def place_cluster(self, comp: LayoutComponent, cluster: list[int], partial: int) -> None:
        """Place a cluster: rho by formula (1), phi from the higher-index neighbours."""
        eps = self.params.epsilon
        gamma = self.params.gamma
        rng = self.rng
        mean = math.pi * len(cluster) / comp.shell_cardinal
        for h in cluster:
            shell_h = self.node_index[h]
            higher: list[tuple[int, float]] = []
            sum_w = 0.0
            sumatory = 0.0
            for w, data in self.graph[h].items():
                if self.node_index[w] > shell_h:
                    weight = float(data.get("weight", 1.0)) if self.params.weighted else 1.0
                    higher.append((w, weight))
                    sum_w += weight
                    sumatory += weight * (self.max_index - self.node_index[w] + 1)
            depth = self.max_index - shell_h
            if higher and depth > 0 and sum_w > 0:
                # Unweighted: sum / (L * depth); weighted: the C++ also divides by sum_w
                divisor = len(higher) * depth * (sum_w if self.params.weighted else 1.0)
                average = sumatory / divisor
            else:
                average = rng.random()

            if self.params.no_cliques and shell_h == self.max_index:
                rho = comp.ratio * rng.random()
                phi = TWO_PI * rng.random()
            else:
                rho = comp.ratio * (1.0 - eps) + eps * comp.ratio * average
                if not self.params.no_cliques:
                    ang_init = TWO_PI * rng.random()
                    phi = 0.0
                    amount = 0.0
                    # The C++ coin-flips every neighbour to the front or back of a list
                    # and then keeps the higher ones; flipping only the higher ones gives
                    # the same distribution of their relative order
                    order = _random_order(range(len(higher)), rng)
                    for i in order:
                        w, weight = higher[i]
                        # The C++ truncated the weighted share to an int (always 0 unless
                        # there is a single neighbour); the float share is used instead
                        new_amount = (
                            weight / sum_w
                            if self.params.weighted
                            else float(self.node_index[w] + 1 - shell_h)
                        )
                        # With an edge index below the node index (k-dense, d-cores) a
                        # higher neighbour can sit in another branch that is placed later;
                        # the C++ read a zero position for it, here it is skipped
                        if new_amount != 0 and w in self.positions:
                            wx, wy = self.positions[w]
                            angle = math.atan2(wy - comp.y, wx - comp.x) + ang_init
                            phi = circular_average(phi, amount, angle, new_amount)
                            amount += new_amount
                    phi -= ang_init
                    if amount == 0:
                        phi = TWO_PI * rng.random()
                else:
                    # Formula (2): the cluster's own angular sector plus normal noise
                    phi = TWO_PI * partial / comp.shell_cardinal + rng.normal(mean, mean)

            # Formulas (6) and (7)
            self.positions[h] = (
                comp.x + gamma * comp.u * rho * math.cos(phi),
                comp.y + gamma * comp.u * rho * math.sin(phi),
            )

    def place_cliques(self, comp: LayoutComponent, cluster: list[int]) -> None:
        """Place a top core: cliques on U-shaped paths in angular sectors."""
        gamma = self.params.gamma
        cliques = _greedy_cliques(cluster, self.graph)
        total = len(cluster)
        hosts_sum = 0
        for clique in cliques:
            random = self.rng.random()
            initial_angle = hosts_sum / total * TWO_PI
            angle = len(clique) / total * TWO_PI
            for i, h in enumerate(clique):
                rho_s, phi_s = place_in_circular_sector(
                    0.90 * comp.ratio, angle, i, len(clique), random, len(clique) == total
                )
                x = rho_s * math.cos(phi_s + initial_angle) + 0.1 * comp.ratio * math.cos(
                    initial_angle + angle / 2.0
                )
                y = rho_s * math.sin(phi_s + initial_angle) + 0.1 * comp.ratio * math.sin(
                    initial_angle + angle / 2.0
                )
                rho = math.hypot(x, y)
                phi = math.atan2(y, x)
                self.positions[h] = (
                    comp.x + gamma * comp.u * rho * math.cos(phi),
                    comp.y + gamma * comp.u * rho * math.sin(phi),
                )
            hosts_sum += len(clique)


def compute_lanet_layout(
    graph: nx.Graph,
    node_index: dict[int, int],
    params: LayoutParameters,
    seed: int = 0,
    edge_index: Callable[[int, int], int] | None = None,
) -> LanetLayout:
    """
    Place every node with the LaNet-vi algorithm.

    Parameters
    ----------
    graph : nx.Graph
        The network (undirected view is used for neighbourhoods)
    node_index : Dict[int, int]
        Shell / dense index of every node
    params : LayoutParameters
        ``epsilon``, ``delta``, ``gamma``, ``u``, ``no_cliques``, ``weighted``
    seed : int
        Seed of the random generator (the C++ ``-seed``)
    edge_index : Callable[[int, int], int], optional
        Index of an edge; defaults to the minimum of its endpoints' indices (k-cores)

    Returns
    -------
    LanetLayout
        Node positions, the component tree (centres, radii, scales) and the frame size
    """
    if graph.is_directed():
        graph = graph.to_undirected(as_view=True)
    if edge_index is None:

        def edge_index(u: int, v: int) -> int:
            return min(node_index[u], node_index[v])

    rng = np.random.default_rng(seed)
    root = build_component_tree(graph, node_index, edge_index, rng)
    root.u = params.u
    placer = _Placer(graph, node_index, params, rng)
    if graph.number_of_nodes():
        placer.radii(root)
        placer.place(root)

    # The C++ frames the picture on the outermost component that has nodes of its own
    outer = root
    min_index = min(node_index.values()) if node_index else 0
    while outer.index != min_index and outer.children:
        outer = outer.children[0]
    frame = params.gamma * params.u * outer.ratio
    return LanetLayout(positions=placer.positions, root=root, frame=frame)


def node_radius(
    degree: int,
    max_degree: int,
    strength: float = 0.0,
    max_strength: float = 0.0,
    weighted: bool = False,
) -> float:
    """Node radius in layout units (``computeHostRatio``, classic mode)."""
    if weighted:
        if max_strength <= 1.0:
            return 0.4
        return 0.4 * math.log(1.0 + strength) / math.log(max_strength)
    if max_degree <= 1:
        return 0.4  # the C++ would divide by log(1) = 0
    return float(0.4 * (math.log(1 + degree) / math.log(max_degree)) ** 0.7)
