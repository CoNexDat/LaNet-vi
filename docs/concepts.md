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
- **k=60-79** (inner): Tier-1 backbone providers (Level3, Telia) and Hypergiant networks (Google, AWS, Cloudflare)

The visualization clearly shows this hierarchical structure with colored concentric rings.

## K-Dense Decomposition

LaNet-vi also supports **k-dense decomposition** (also called m-core):

- Based on **triangle density** rather than degree
- A k-dense subgraph requires k triangles per node
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
