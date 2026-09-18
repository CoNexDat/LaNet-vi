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
- **Scalability**: Works on networks with millions of nodes

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

## Directed Cores (D-Cores)

For directed networks, LaNet-vi computes **d-cores**:

- Considers **in-degree** and **out-degree** separately
- Node coreness: (k_in, k_out) tuple
- Useful for citation networks, web graphs, and social media

## References

- Alvarez-Hamelin, J.I., Dall'Asta, L., Barrat, A., Vespignani, A. (2005). "k-core decomposition: a tool for the visualization of large scale networks". *arXiv preprint*.

- Seidman, S.B. (1983). "Network structure and minimum degree". *Social Networks*, 5(3), 269-287.
