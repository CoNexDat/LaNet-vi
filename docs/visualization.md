# Visualization Guide

## How LaNet-vi Plots Work

LaNet-vi creates hierarchical network visualizations using a circular, shell-based layout that reveals the k-core structure.

## Layout Algorithm

The placement is the one of the C++ LaNet-vi (Alvarez-Hamelin, Dall'Asta, Barrat &
Vespignani, NIPS 2005; `-coordDistributionAlgorithm classic`):

1. **Nested components.** Inside the component of index *k* (all its nodes have index
   ≥ *k*), the connected pieces of the nodes with index > *k* become child components of
   index *k* + 1; the nodes with index exactly *k* form the component's *clusters*. The
   whole graph is the root component of index 0, so every k-core, k-dense or d-core
   level is a layer of this tree.
2. **Radii.** The top core of a branch gets a disc whose radius grows with the square
   root of the sum of the squared log-degrees of its nodes; each enclosing shell adds one
   unit of radius. Shells are therefore rings one unit apart, the highest index innermost.
3. **Centres.** A component with siblings is offset from its parent's centre by
   `rho = 1 - size / siblings` (a lone child is concentric), at an angle that grows with
   the cumulative size of the siblings before it, and drawn at a smaller scale
   `u = sqrt(size / siblings) / delta`.
4. **Nodes.** A node of index *k* sits at `rho = R (1 - epsilon) + epsilon R avg`, where
   `avg` measures how deep its higher-index neighbours are (closer to the centre when
   they are deep), and at the circular average of the angles of those neighbours, which
   are placed first. Top cores are split into cliques, each laid along a U-shaped path in
   its own angular sector. `--no-cliques` spreads top cores uniformly and gives every
   cluster its own sector instead (formula (2) of the paper).
5. `gamma` scales the whole picture; `--draw-circles` draws every component's disc.

Node radius is `0.4 (log(1 + d) / log(d_max))^0.7` layout units (strength-based for
weighted graphs), never less than one pixel; `--node-size-scale` multiplies it.

**Parameters:**
- `epsilon` (0.18): thickness of each ring as a fraction of its radius
- `delta` (1.3): how much smaller sibling components are drawn
- `gamma` (1.5): component diameter / picture scale
- `unit_length` (1.0): the root scale `u`
- `seed`: random seed (cluster order, ties, the random angle frame)

The `pow`/`log` coordinate distributions of the C++ (circle packing of siblings) are not
ported yet (#18).

## Node Visualization

### Node Color

Nodes are colored by their k-core number using a color scheme:

- **Low k-core** (periphery): Blue/purple colors
- **Medium k-core**: Green/yellow colors
- **High k-core** (core): Red/orange colors

Color schemes:
- `color` (default): Full spectrum from blue→green→yellow→red
- `bw`: Grayscale from black→white

### Node Size

Node size represents **degree** (number of connections):

- Larger nodes = Higher degree (more connections)
- Smaller nodes = Lower degree (fewer connections)

Size scaling:
- Logarithmic for large networks (prevents huge nodes)
- Linear for small networks
- Configurable via `node_size_scale` parameter

## Edge Visualization

### Gradient Edge Coloring

Edges use a **gradient color** that blends the colors of both endpoints:

```
Node A (k=10, red) ────────── Node B (k=3, blue)
           red ──→ purple ──→ blue
```

**How it works:**
1. Each edge is split into two segments at the midpoint
2. First half: colored by source node's k-core
3. Second half: colored by target node's k-core
4. Creates smooth gradient transition

This visualization technique shows:
- **High-k to high-k edges**: Bright colors in center
- **Low-k to low-k edges**: Cool colors at periphery
- **Cross-shell edges**: Visible color gradients

### Edge Filtering

For large networks, only a subset of edges is drawn:

- `edges_percent`: Percentage of total edges to show (0.0-1.0)
- `min_edges`: Minimum number of edges (overrides percentage)

**Stratified sampling**: Edges are sampled proportionally from all k-shells to maintain structure visibility.

### Edge Styling

- `edge_alpha`: Transparency (0.0 = invisible, 1.0 = opaque)
- `min_edge_width` / `max_edge_width`: Width range
- `gradient_edges`: Enable/disable gradient coloring

## Component Circles

Optional component border circles:

- Drawn around connected components
- Useful for fragmented networks
- Controlled by `draw_circles` parameter

## Legends

### K-Core Legend (Right Side)

Shows the color mapping for k-core values:

```
k-core
  79  ●  (red)
  66  ●  (orange)
  53  ●  (yellow)
  40  ●  (green)
  27  ●  (cyan)
  14  ●  (blue)
   1  ●  (purple)
```

- Position: center right
- Smaller circles for compact legend
- Selective labeling (not all values shown)

### Degree Legend (Left Side)

Shows node size scale:

```
degree
  6241  ●  (large)
  3120  ●  (medium)
  1560  ●  (small)
     1  ●  (tiny)
```

- Position: upper left
- Circle sizes match visualization
- Shows degree range in network

### Legend Configuration

- `show_color_legend`: Show/hide the shell/dense index colour legend
- `show_degree_scale`: Show/hide the degree (node size) legend, as the C++ `-showDegreeScale`
  (`show_size_legend` is a deprecated alias)
- `legend_fontsize`: Manual font size (or auto-scales with diagram)
- Text color: Automatic (white on dark, black on light)

## Background and Colors

- `background`: `black` (default) or `white`
- `color_scheme`: `col` (color) or `bw` (black & white)

**Recommendation**: Black background with color scheme for large networks (better contrast).

## Example Interpretation

### CAIDA AS-Relationships Visualization

Looking at a typical output:

**Center (red/orange):**
- High k-core (k≈100-149 in the 2025 CAIDA snapshot)
- Tier-1 ISPs and backbone providers
- Dense interconnection
- Few nodes, many edges between them

**Middle rings (yellow/green):**
- Medium k-core (k=20-50)
- Regional ISPs and medium providers
- Moderate connectivity
- Transitional layer

**Outer rings (blue/purple):**
- Low k-core (k=1-20)
- Stub networks and end users
- Sparse connections
- Many nodes, few connections each

**Edges:**
- Bright inner edges: Backbone interconnections
- Gradient edges: Provider-customer relationships
- Outer sparse edges: Access connections

## Customization Tips

### For Large Networks (>10K nodes)

```python
config.visualization.edges_percent = 0.1  # Show 10% of edges
config.visualization.edge_alpha = 0.5     # Semi-transparent
config.visualization.node_size_scale = 0.4  # Smaller nodes
config.visualization.min_edge_width = 0.03  # Thinner edges
```

### For Small Networks (<1K nodes)

```python
config.visualization.edges_percent = 1.0  # Show all edges
config.visualization.edge_alpha = 0.8     # More opaque
config.visualization.node_size_scale = 2.0  # Larger nodes
config.visualization.epsilon = 0.3        # Thicker rings
```

### For Publication-Quality

```python
config.visualization.width = 3600   # High resolution
config.visualization.height = 3600
config.visualization.background = "white"
config.visualization.edge_alpha = 0.6
```

## Spiral Layout

`config.layout.use_spiral_layout` and the `spiral_*` settings are accepted but not
implemented (#18); the classic placement is always used.
