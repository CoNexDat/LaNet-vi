# K-Core Decomposition Concepts

## What is K-Core Decomposition?

K-core decomposition is a method for analyzing the hierarchical structure of networks by identifying increasingly dense subgraphs.

### Definition

A **k-core** is the maximal subgraph in which every node has at least **k** neighbors within that subgraph.

- The **k-shell** consists of all nodes with coreness exactly k (in the k-core but not in the (k+1)-core)
- The **k-core number** (or coreness) of a node is the highest k for which that node belongs to a k-core

### Visual Representation

```
Network with k-core structure:

     Outermost Shell (k=1)
    ○────────────────────○
   ○    ○─────────○     ○
  ○    Middle (k=2)    ○
  ○   ○──────────○    ○
   ○   ○ Core  ○    ○
    ○  ○ (k=3)○   ○
     ○  ○───○   ○
      ○  ○─○  ○
       ○──○─○

Concentric structure:
┌─────────────────────────┐
│  k=1 (periphery)        │
│  ┌───────────────────┐  │
│  │  k=2 (middle)     │  │
│  │  ┌─────────────┐  │  │
│  │  │  k=3 (core) │  │  │
│  │  │     ●●●●    │  │  │
│  │  │    ●●●●●    │  │  │
│  │  │     ●●●●    │  │  │
│  │  └─────────────┘  │  │
│  └───────────────────┘  │
└─────────────────────────┘
```

### The K-Core Decomposition Algorithm

1. **Find k=1**: Remove all nodes with degree 0
2. **Find k=2**: Remove all nodes with degree ≤ 1 (iteratively, as removal changes degrees)
3. **Find k=3**: Remove all nodes with degree ≤ 2
4. **Continue** until all nodes are assigned a k-core number

This creates a hierarchical "onion-like" structure where:
- **Outer layers** (low k): Peripheral nodes, loosely connected
- **Middle layers**: Intermediate connectivity
- **Inner core** (high k): Densely connected central nodes

### Why K-Core Decomposition?

**Advantages:**
- **Fast computation**: O(|E|) time complexity
- **Hierarchical view**: Natural layers reveal network organization
- **Resilience measure**: High k-core nodes are more central and important
- **Scalability**: the decomposition itself is linear; LaNet-vi draws networks of
  hundreds of thousands of nodes

**Applications:**
- **Social networks**: Identify influential users and communities
- **Internet topology**: Understand AS-level structure (backbone vs edge networks)
- **Biological networks**: Find functional modules in protein interaction networks
- **Collaboration networks**: Identify core research groups

### Example: Internet AS-Level Topology

In the CAIDA AS-relationships dataset:
- **k=1-10** (outer): Stub networks, small ISPs, end users
- **k=20-40** (middle): Regional ISPs, medium providers
- **k≈100-149** (inner, 2025 snapshot): Tier-1 backbone providers (Level3, Telia) and Hypergiant networks (Google, AWS, Cloudflare)

The visualization clearly shows this hierarchical structure with colored concentric rings.

## Weighted K-Cores

With `--weighted`, the degree is replaced by the **strength** (sum of incident edge
weights) and the integer shells by strength intervals, as in the C++ LaNet-vi:

- A *p-function* splits the strength range into `--granularity` intervals (default:
  as many as the maximum degree). `--strength-intervals` chooses how:
  `equalIntervalSize` (default; equal width up to the largest strength or
  `--maximum-strength`), `equalNodesPerInterval` (boundaries taken from the sorted
  strengths so every interval holds the same number of nodes), `equalLogIntervalSize`
  (geometric progression) or `custom` (boundaries read one per line from
  `--strength-intervals-file`)
- Every node starts in the interval of its total strength, then shells are peeled in
  increasing order: when a node of shell k is removed, each neighbor is re-binned
  using only the strength it still receives from nodes above shell k, and never moves
  below k. A node has index ≥ k iff it belongs to a subgraph where every member
  receives strength in interval ≥ k from the other members — the generalized k-core
- Indices run 1..granularity (0 for an isolated node); with `custom` there is one
  index per boundary in the file instead. The C++ 3.0.1 numbered two of the interval
  methods 2..granularity+1 because of a duplicated 0.0 boundary
- `--maximum-strength` fixes the top boundary of `equalIntervalSize` and
  `equalLogIntervalSize` so pictures of different networks share the same scale
  (`equalNodesPerInterval` takes its boundaries from the data and `custom` from the file)

## K-Dense Decomposition

LaNet-vi also supports **k-dense decomposition** (also called m-core):

- Based on **triangles** rather than degree
- The k-dense of a graph is the maximal subgraph in which every edge closes at least
  k − 2 triangles inside the subgraph (the k-truss); the m-core numbering counts the
  triangles, m = k − 2
- Every edge is assigned the largest k whose k-dense contains it (2 for an edge in no
  triangle), and a node takes the largest index among its edges
- Computed by peeling edges in order of triangle count, exactly as the C++ LaNet-vi
  (removing an edge lowers the count of the two other sides of each triangle it closed)
- Identifies more cohesive structures than k-cores
- Useful for community detection and clustering analysis

## K-Connectivity

The k-core index bounds the degree, not the connectivity: a node of the k-core has at
least k neighbors in it, but k edge-disjoint paths to the rest of the core are not
guaranteed. Beiró, Alvarez-Hamelin & Busch (2008) derived a lower bound of the edge
connectivity from the picture itself (`--kconn`, the C++ `-kconn`):

- The nodes of each shell are split into **clusters**, the connected pieces of the shell
  inside a component (the same clusters the layout draws).
- The walk starts at the top: the first cluster whose induced subgraph has diameter at
  most 2 (or, in the `wide` variant, a minimum edge cut of at least its shell index; in
  `strict`, a minimum degree of at least the index) seeds the **k-connected set** `C`.
- Going down shell by shell, a cluster `Q` of shell `k` joins `C` with k-connectivity
  `k` when `Q` with `C` contracted to one vertex has diameter at most 2 and either at
  least `k` of its nodes touch `C`, or all of them do, or the bound
  `phi(Q) = sum(min(max(1, |N(v) ∩ NotB2|), |N(v) ∩ C|))` reaches `k`, where `NotB2`
  are the nodes of `Q` with fewer than two neighbors in `C`.
- `wide` (the default) keeps the clusters it skipped and tries them again at every
  lower index, so a node can be k-connected for a `k` below its shell index; `strict`
  drops them, so its values are either 0 or the shell index.

Nodes that never join `C` are not k-connected; the picture paints them black on white /
white on black (squares in the grayscale schemes). The values are those the C++ tool
computed, including its order effects: clusters are examined once per shell in the
order of the component tree, so a cluster rejected before its neighbors joined `C` may
stay out (the strict variant finds no seed at all when the top core has diameter 3, as
in the karate club).

## Directed Cores (D-Cores)

For directed networks (`--directed --decomp dcores`), LaNet-vi computes **d-cores**
(Giatsidis, Thilikos & Vazirgiannis, 2011):

- In-degree and out-degree are peeled separately: `k_in` is the largest k such that the
  node belongs to the subgraph where every node has in-degree ≥ k, and `k_out` the same
  for the out-degree
- Every node gets the pair `(k_in, k_out)` (exported in the JSON output and kept in
  `result.metadata["d_cores"]`); the picture places it by `max(k_in, k_out)`
- The full **(k, l)-core table** of the paper is available with `--dcore-table FILE`
  (`compute_dcore_table` in Python): the (k, l)-core is the largest subgraph in which
  every node has in-degree ≥ k *and* out-degree ≥ l, and the table gives, for every l,
  the largest k such that each node of the (0, l)-core is in the (k, l)-core. Row 0 is
  `k_in`. This is what the C++ 4.0.0 tool computed for a directed graph (its
  `dcores_list.txt`); that code kept a node in the in-degree peeling after its
  out-degree had dropped below l and so reported a larger k for such nodes, while
  LaNet-vi 5 follows the definition
- Useful for citation networks, web graphs and follower networks, where being cited and
  citing are different roles

## References

- Seidman, S.B. (1983). "Network structure and minimum degree". *Social Networks*, 5(3), 269–287.
- Alvarez-Hamelin, J.I., Dall'Asta, L., Barrat, A., Vespignani, A. (2006). "Large scale networks fingerprinting and visualization using the k-core decomposition". *Advances in Neural Information Processing Systems 18* (NIPS 2005). Also as "k-core decomposition: a tool for the visualization of large scale networks", arXiv:cs/0504107.
- Beiró, M.G., Alvarez-Hamelin, J.I., Busch, J.R. (2008). "A low complexity visualization tool that helps to perform complex systems analysis". *New Journal of Physics*, 10, 125003.
- Giatsidis, C., Thilikos, D.M., Vazirgiannis, M. (2011). "D-cores: measuring collaboration of directed graphs based on degeneracy". *IEEE ICDM 2011*.
- Cohen, J. (2008). "Trusses: cohesive subgraphs for social network analysis". *National Security Agency technical report* (the k-truss, which the k-dense decomposition computes).
