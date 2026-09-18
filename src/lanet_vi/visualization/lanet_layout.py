"""The LaNet-vi placement: nested components, neighbor-based rho, clique sectors.

Port of ``kcores_component.cpp`` / ``graph_kcores_components.cpp`` from the C++ LaNet-vi
(``-coordDistributionAlgorithm classic``, the default), following the formulas of
Alvarez-Hamelin, Dall'Asta, Barrat & Vespignani, *k-core decomposition: a tool for the
visualization of large scale networks* (NIPS 2005):

1. The graph is split recursively into nested **components**: the connected components of
   the subgraph with index ``> k`` inside a component of index ``k``. A component of index
   ``k`` owns the **clusters** of its nodes with index exactly ``k`` (connected inside the
   shell) and its children (index ``k + 1``).
2. Each component gets a center, a radius (``ratio``) and a scale (``u``): the top core of
   a branch has a radius proportional to the root of the sum of its squared log-degrees,
   every enclosing shell adds one unit (part 1); children are placed inside their parent
   at ``rho = 1 - size / siblings``, ``phi = 2 pi (previous / siblings)^2`` and scaled by
   ``sqrt(size / siblings) / delta`` (formulas (3)-(5), part 2).
3. A node of index ``k`` sits at ``rho = ratio (1 - eps) + eps ratio * average``, where
   ``average`` measures how deep its higher-index neighbors are (formula (1)), and at the
   circular average of the angles of those neighbors (already placed). Top cores are
   split into cliques laid along U-shaped paths in angular sectors (formula (2)).

K-dense and d-core results go through the same placement with their own *edge index*
for the component tree (an edge belongs to the inner component when its index is above
the component's: for k-cores the minimum of its endpoints' indices, for k-dense the
edge's own dense index, as ``kdenses_component.cpp`` walks it).

The ``pow`` and ``log`` coordinate distributions (``findCoordinatesModern``) replace step
2: the whole network is a disc of radius 1; inside a component of radius ``ratio`` the
children share a disc of radius ``R`` (``(sqrt((T - S) / T))^0.25 ratio`` capped at
``0.96 ratio`` for ``pow``, ``sqrt((size - shell) / size) ratio`` for ``log``, with ``T``
and ``S`` the sums of squared log-degrees of the component and of its own shell), packed
as non-overlapping discs whose areas follow their ``sum(log(1 + d)^(2 / beta))`` weight
(:func:`lanet_vi.visualization.layout.distribute_components`); the nodes of the shell
sit on the ring between ``R`` and ``ratio`` by formula (1). The k-dense variant of
``kdenses_component.cpp`` (the only placement the C++ had for k-dense) shrinks by a fixed
``0.92``, counts the neighbors of the same index in formula (1), scales ``epsilon`` by
``tau = (ratio - R) / ratio`` and auto-adjusts the ``ratioConstant`` of the node radii.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from lanet_vi.visualization.layout import distribute_components

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
    #: ``pow`` / ``log`` modes: radius of the disc holding the children (``endRatio``,
    #: 0 for a top core) and the log-degree sums of ``computeComponents``
    end_ratio: float = 0.0
    t_log2_degree: float = 0.0
    shell_t_log2_degree: float = 0.0
    t_log_beta_degree: float = 0.0

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
    #: ``classic`` (default), ``pow`` or ``log`` (``-coordDistributionAlgorithm``)
    coord_distribution: str = "classic"
    #: Constant and exponent of the disc area law of the circle packing (``-alpha``,
    #: ``-beta``; the C++ defaults)
    alpha: float = 0.3
    beta: float = 1.0
    #: Use the k-dense variant of ``kdenses_component.cpp`` in ``pow`` / ``log`` mode
    dense: bool = False
    #: ``-ratioConstant``: node radius factor of the ``pow`` / ``log`` modes; ``None``
    #: is the C++ "auto-adjusted" (1 for k-cores, the top-core rule for k-dense)
    ratio_constant: float | None = None

    @property
    def modern(self) -> bool:
        """Whether the ``pow`` / ``log`` placement is used instead of ``classic``."""
        return self.coord_distribution != "classic"


@dataclass
class LanetLayout:
    """Result of :func:`compute_lanet_layout`."""

    positions: dict[int, tuple[float, float]]
    root: LayoutComponent
    #: The network radius on the picture, ``gamma * u * R`` with ``R`` the radius of the
    #: outermost non-empty component (1 in the ``pow`` / ``log`` modes). The C++ viewport
    #: (``svg.cpp``) spans 1.6 times this horizontally and 1.2 times vertically; the
    #: legends sit in that margin.
    frame: float
    #: The effective ``ratioConstant`` of the node radii (``pow`` / ``log`` modes)
    ratio_constant: float = 1.0


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


def _merge_parallel_edges(graph: nx.MultiGraph) -> nx.Graph:
    """Collapse parallel edges into one, summing their weights (the C++ strength)."""
    simple = nx.Graph()
    simple.add_nodes_from(graph.nodes())
    for u, v, data in graph.edges(data=True):
        w = float(data.get("weight", 1.0))
        if simple.has_edge(u, v):
            simple[u][v]["weight"] += w
        else:
            simple.add_edge(u, v, weight=w)
    return simple


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
    neighbors, in adjacency order, when they are adjacent to every member; cliques are
    sorted by node id. The C++ meant to rank nodes by the connections among their
    top-core neighbors, but it looked the core numbers up in the empty map of a fresh
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
    """Runs ``findCoordinatesClassic`` (two passes) or ``findCoordinatesModern`` over a tree."""

    def __init__(
        self,
        graph: nx.Graph,
        node_index: dict[int, int],
        params: LayoutParameters,
        rng: np.random.Generator,
        degrees: dict[int, int] | None = None,
        seed: int = 0,
    ) -> None:
        self.graph = graph
        self.node_index = node_index
        self.params = params
        self.rng = rng
        self.seed = seed
        # The k-dense variant of the pow / log modes (kdenses_component.cpp)
        self.dense_modern = params.modern and params.dense
        # -ratioConstant starts at 1 and, for k-dense, the top cores may lower it
        self.ratio_constant = 1.0 if params.ratio_constant is None else params.ratio_constant
        self.max_index = max(node_index.values()) if node_index else 0
        self.degree = dict(graph.degree()) if degrees is None else degrees
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

    # -- part 2: centers and node positions ------------------------------------------
    def place(self, root: LayoutComponent) -> None:
        """Center and scale every component, then its nodes once its children are placed.

        Same order as the C++ recursion: a component's center first, then its children
        (depth first), then its own clusters. Iterative for the same reason as ``radii``.
        """
        stack: list[tuple[LayoutComponent, bool]] = [(root, False)]
        while stack:
            comp, children_done = stack.pop()
            if children_done:
                self.place_own_nodes(comp)
                continue
            self.place_center(comp)
            stack.append((comp, True))
            stack.extend((child, False) for child in reversed(comp.children))

    # -- pow / log modes: circle packing ----------------------------------------------
    def log_degree_sums(self, root: LayoutComponent) -> None:
        """Accumulate the ``tLog2Degree`` / ``tLogBetaDegree`` sums of every component."""
        exponent = 2.0 / self.params.beta
        for comp in _post_order(root):
            for cluster in comp.clusters:
                for v in cluster:
                    log_degree = math.log(1 + self.degree[v])
                    comp.shell_t_log2_degree += log_degree**2
                    comp.t_log_beta_degree += log_degree**exponent
            comp.t_log2_degree = comp.shell_t_log2_degree
            for child in comp.children:
                comp.t_log2_degree += child.t_log2_degree
                comp.t_log_beta_degree += child.t_log_beta_degree

    def place_modern(self, root: LayoutComponent) -> None:
        """``findCoordinatesModern``: pack the children of every component, then its nodes.

        Same order as the classic pass 2 (a component's children are placed before its own
        clusters). The root is the unit disc; ``u`` stays the parameter (the modern mode
        never rescales it), and the k-dense variant does not use it at all.
        """
        root.ratio = 1.0
        stack: list[tuple[LayoutComponent, bool]] = [(root, False)]
        while stack:
            comp, children_done = stack.pop()
            if children_done:
                self.place_own_nodes(comp)
                continue
            comp.u = 1.0 if self.dense_modern else self.params.u
            self.distribute_children(comp)
            stack.append((comp, True))
            stack.extend((child, False) for child in reversed(comp.children))

    def distribute_children(self, comp: LayoutComponent) -> None:
        """Give the children their centers and radii inside the disc of radius ``R``."""
        params = self.params
        log_mode = params.coord_distribution == "log"
        if not comp.children:
            # kdenses_component.cpp: a top core lowers -ratioConstant (unless given) to
            # half its radius over the root of its squared log-degree sum
            if self.dense_modern and params.ratio_constant is None and comp.t_log2_degree > 0:
                self.ratio_constant = min(
                    self.ratio_constant, 0.5 * comp.ratio / math.sqrt(comp.t_log2_degree)
                )
            return
        ratio = comp.ratio
        if comp.shell_t_log2_degree != 0:
            if self.dense_modern:
                # The 0.97 is absolute in the C++ (the root has radius 1 anyway)
                radius = 0.92 * ratio if comp.index > 1 else 0.97
            elif log_mode:
                radius = math.sqrt(comp.size - comp.shell_cardinal) / math.sqrt(comp.size) * ratio
            else:
                share = math.sqrt(comp.t_log2_degree - comp.shell_t_log2_degree)
                share /= math.sqrt(comp.t_log2_degree)
                radius = min(share**0.25 * ratio, 0.96 * ratio)
            comp.end_ratio = radius
        else:
            radius = ratio
        if len(comp.children) > 1:
            # Disc areas follow sum(log(1 + d)^(2 / beta)); the k-dense log mode uses the
            # size (the k-core "modernLog" branch that used it was unreachable)
            weights = np.array(
                [
                    float(child.size) if self.dense_modern and log_mode else child.t_log_beta_degree
                    for child in comp.children
                ]
            )
            xs, ys, rs = distribute_components(
                comp.x, comp.y, radius, weights, params.alpha, params.beta, log_mode, self.seed
            )
            for child, x, y, r in zip(comp.children, xs, ys, rs, strict=True):
                child.x, child.y, child.ratio = float(x), float(y), float(r)
        else:
            (child,) = comp.children
            child.x, child.y, child.ratio = comp.x, comp.y, radius

    def place_center(self, comp: LayoutComponent) -> None:
        """Formulas (4), (3) and (5): a child's rho, phi, center and scale in its parent."""
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
                elif self.dense_modern:
                    self.place_cluster_dense(comp, cluster, partial)
                else:
                    self.place_cluster(comp, cluster, partial)
            partial += len(cluster)

    def place_cluster(self, comp: LayoutComponent, cluster: list[int], partial: int) -> None:
        """Place a cluster: rho by formula (1), phi from the higher-index neighbors."""
        eps = self.params.epsilon
        gamma = self.params.gamma
        rng = self.rng
        mean = math.pi * len(cluster) / comp.shell_cardinal
        for h in cluster:
            shell_h = self.node_index[h]
            higher: list[tuple[int, float]] = []
            sum_w = 0.0
            sumatory = 0.0
            # Weights are paired with their neighbor here; the C++ advanced its weight
            # iterator only for qualifying neighbors, so they drifted apart after any
            # non-qualifying one (an undocumented C++ bug, not reproduced)
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
                    # The C++ coin-flips every neighbor to the front or back of a list
                    # and then keeps the higher ones; flipping only the higher ones gives
                    # the same distribution of their relative order
                    order = _random_order(range(len(higher)), rng)
                    for i in order:
                        w, weight = higher[i]
                        # The C++ truncated the weighted share to an int (always 0 unless
                        # there is a single neighbor); the float share is used instead
                        new_amount = (
                            weight / sum_w
                            if self.params.weighted
                            else float(self.node_index[w] + 1 - shell_h)
                        )
                        # With an edge index below the node index (k-dense, d-cores) a
                        # higher neighbor can sit in another branch that is placed later;
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

    def place_cluster_dense(self, comp: LayoutComponent, cluster: list[int], partial: int) -> None:
        """Place a k-dense cluster (``kdenses_component.cpp`` ``findClusterCoordinates``).

        Differences with :meth:`place_cluster`: neighbors of the same index count in
        formula (1) and in the angle (the C++ let them in once any component had been
        placed, reading a zero position for the ones still pending; here a neighbor counts
        once it is placed, as for the higher ones); ``epsilon`` is scaled by ``tau``, the
        share of the radius left outside the children's disc; no random rotation of the
        frame; weights are ignored; and the position is ``gamma * rho``, without ``u``.
        A top core reaches this only with ``no_cliques``, where its nodes are spread at
        random; the C++ still computed ``average`` for them, dividing by a zero depth
        (``inf``, unused), which the ``depth > 0`` guard skips.
        """
        eps = self.params.epsilon
        gamma = self.params.gamma
        rng = self.rng
        mean = math.pi * len(cluster) / comp.shell_cardinal
        top = comp.end_ratio == 0.0
        tau = (comp.ratio - comp.end_ratio) / comp.ratio
        for h in cluster:
            dense_h = self.node_index[h]
            same_or_higher = [w for w in self.graph[h] if self.node_index[w] >= dense_h]
            depth = self.max_index - dense_h
            if same_or_higher and depth > 0:
                sumatory = sum(self.max_index - self.node_index[w] + 1 for w in same_or_higher)
                average = sumatory / (len(same_or_higher) * depth)
            else:
                average = rng.random()

            if self.params.no_cliques and top:
                rho = comp.ratio * rng.random()
                phi = TWO_PI * rng.random()
            else:
                rho = comp.ratio * (1.0 - eps * tau) + eps * tau * comp.ratio * average
                if not self.params.no_cliques or not top:
                    phi = 0.0
                    amount = 0.0
                    for i in _random_order(range(len(same_or_higher)), rng):
                        w = same_or_higher[i]
                        if w in self.positions:
                            new_amount = float(self.node_index[w] + 1 - dense_h)
                            wx, wy = self.positions[w]
                            angle = math.atan2(wy - comp.y, wx - comp.x)
                            phi = circular_average(phi, amount, angle, new_amount)
                            amount += new_amount
                    if amount == 0:
                        phi = TWO_PI * rng.random()
                else:
                    phi = TWO_PI * partial / comp.shell_cardinal + rng.normal(mean, mean)

            self.positions[h] = (
                comp.x + gamma * rho * math.cos(phi),
                comp.y + gamma * rho * math.sin(phi),
            )

    def place_cliques(self, comp: LayoutComponent, cluster: list[int]) -> None:
        """Place a top core: cliques on U-shaped paths in angular sectors.

        The k-dense variant drops the ``0.1 ratio`` offset towards the sector's middle
        and the ``u`` factor.
        """
        gamma = self.params.gamma
        offset = 0.0 if self.dense_modern else 0.1
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
                x = rho_s * math.cos(phi_s + initial_angle) + offset * comp.ratio * math.cos(
                    initial_angle + angle / 2.0
                )
                y = rho_s * math.sin(phi_s + initial_angle) + offset * comp.ratio * math.sin(
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
        The network (undirected view is used for neighborhoods)
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
        Node positions, the component tree (centers, radii, scales) and the frame size
    """
    # Degrees as the C++ getDegree(): neighbors with multiplicity (in + out when directed)
    degrees = dict(graph.degree())
    if graph.is_directed():
        graph = graph.to_undirected(as_view=True)
    if graph.is_multigraph():
        graph = _merge_parallel_edges(graph)
    if edge_index is None:

        def edge_index(u: int, v: int) -> int:
            return min(node_index[u], node_index[v])

    rng = np.random.default_rng(seed)
    root = build_component_tree(graph, node_index, edge_index, rng)
    root.u = params.u
    placer = _Placer(graph, node_index, params, rng, degrees, seed=seed)
    if graph.number_of_nodes():
        if params.modern:
            placer.log_degree_sums(root)
            placer.place_modern(root)
        else:
            placer.radii(root)
            placer.place(root)

    if params.modern:
        # generateNetworkFile: the pow / log picture is the unit disc
        frame = params.gamma * params.u
    else:
        # The C++ frames the classic picture on the outermost component with nodes of its own
        outer = root
        min_index = min(node_index.values()) if node_index else 0
        while outer.index != min_index and outer.children:
            outer = outer.children[0]
        frame = params.gamma * params.u * outer.ratio
    return LanetLayout(
        positions=placer.positions,
        root=root,
        frame=frame,
        ratio_constant=placer.ratio_constant,
    )


@dataclass(frozen=True)
class RadiusLaw:
    """Node radius in layout units for the active placement (``computeHostRatio``).

    ``classic`` uses :func:`node_radius`. The ``pow`` / ``log`` modes use
    ``0.007 ratioConstant log(1 + d)^1.5`` (``graphics_kcores.cpp``), or
    ``ratioConstant sqrt(log(1 + d))`` for k-dense (``graphics_kdenses.cpp``, whose
    radii only ever went with that placement); on weighted graphs both use
    ``ratioConstant log(1 + s) / log(s_max)`` (times 0.007 for k-cores), with the same
    fallback to the degree law as :func:`strength_radii`.
    """

    max_degree: int
    max_strength: float = 0.0
    weighted: bool = False
    modern: bool = False
    dense: bool = False
    ratio_constant: float = 1.0

    def __call__(self, degree: int, strength: float = 0.0, weighted: bool | None = None) -> float:
        """Radius of a node of the given degree (and strength on weighted graphs).

        ``weighted=False`` forces the degree law, as the C++ edge widths do.
        """
        weighted = self.weighted if weighted is None else weighted
        if not self.modern:
            return node_radius(degree, self.max_degree, strength, self.max_strength, weighted)
        factor = self.ratio_constant if self.dense else 0.007 * self.ratio_constant
        if strength_radii(weighted, self.max_strength):
            return factor * math.log(1.0 + strength) / math.log(self.max_strength)
        if self.dense:
            return factor * math.sqrt(math.log(1 + degree))
        return float(factor * math.log(1 + degree) ** 1.5)


def strength_radii(weighted: bool, max_strength: float) -> bool:
    """Whether node radii follow the strength law (weighted and a usable maximum)."""
    return weighted and max_strength > 1.0


def node_radius(
    degree: int,
    max_degree: int,
    strength: float = 0.0,
    max_strength: float = 0.0,
    weighted: bool = False,
) -> float:
    """Node radius in layout units (``computeHostRatio``, classic mode).

    The strength law needs ``log(max_strength) > 0``; when every strength is at most 1
    (the C++ would divide by zero or a negative number) the degree law is used instead,
    see :func:`strength_radii`.
    """
    if strength_radii(weighted, max_strength):
        return 0.4 * math.log(1.0 + strength) / math.log(max_strength)
    if max_degree <= 1:
        return 0.4  # the C++ would divide by log(1) = 0
    return float(0.4 * (math.log(1 + degree) / math.log(max_degree)) ** 0.7)
