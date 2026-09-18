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
3. **Centers.** A component with siblings is offset from its parent's center by
   `rho = 1 - size / siblings` (a lone child is concentric), at an angle that grows with
   the cumulative size of the siblings before it, and drawn at a smaller scale
   `u = sqrt(size / siblings) / delta`.
4. **Nodes.** A node of index *k* sits at `rho = R (1 - epsilon) + epsilon R avg`, where
   `avg` measures how deep its higher-index neighbors are (closer to the center when
   they are deep), and at the circular average of the angles of those neighbors, which
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

K-dense and d-core pictures use this same placement, with the component tree built from
the edge dense index (k-dense) or the minimum endpoint index (d-cores). The `pow`/`log`
coordinate distributions of the C++ (circle packing of siblings, and with them the
k-dense specific variant of `kdenses_component.cpp`) are not ported yet (#18).

## Node Visualization

### Node Color

Nodes are colored by their shell (or dense) index with the color scale of the C++
LaNet-vi (`types.cpp`):

- `col` (default): magenta → blue → cyan → green → yellow → **red** from the periphery to
  the maximum index. Consecutive shells alternate a luminosity of 0.7 and 1.2 so that
  neighboring rings stay distinguishable (k-dense pictures on a white background use a
  constant 0.9 instead).
- `bw`: white → gray → **black** from the periphery to the maximum index.
- `bwi`: the same scale interlaced: indices with the parity of the maximum take the dark
  half, the others the light half, so adjacent shells contrast strongly.

`color_scale_max_value` (`--color-scale-max`) fixes the index drawn with the last color;
higher indices share it, which makes pictures of different networks comparable. For k-dense
pictures with `measure = mcore` (the default) the value is an m-core number (k-dense
minus 2), as in the C++. A network with a single shell is red.

A colors file (`--colors-file`) replaces the shell colors; nodes absent from it are white
on a black background and black on a white one, and the color legend is not drawn.

### Node Size

Node radius follows the C++ `computeHostRatio` in layout units: `0.4 (log(1+d) /
log(dmax))^0.7` for degree `d` (a strength-based law on weighted graphs), never smaller
than one pixel. `node_size_scale` multiplies it (1.0 is the C++ size).

## Edge Visualization

### Gradient Edge Coloring

Each edge is drawn as two halves meeting at the midpoint, as the two cylinders of the C++:
the half next to node A takes the color of node B and vice versa, darkened by 0.75 in
`col` pictures and lightened by 1.2 in `bw`/`bwi` ones. K-dense edges are one color, the
color of their own dense index, darkened by 0.5. `gradient_edges: false` draws plain
edges in the text color instead.

Edges are drawn under the nodes in increasing index order, so the edges of the core end up
on top of the peripheral ones.

### Edge Width

The width of an edge is 0.2 host radii of the smaller endpoint degree
(`0.2 * 0.4 (log(1+d) / log(dmax))^0.7` layout units, the C++ `ratioEdge`), never thinner
than one pixel. `min_edge_width` / `max_edge_width` are deprecated and have no effect.

### Edge Filtering

For large networks, only a subset of edges is drawn: every edge is kept independently with
probability `max(edges_percent, min_edges / E)`, the C++ Bernoulli draw, seeded with the
layout `seed` so the same seed gives the same picture.

- `edges_percent`: fraction of the edges to show (0.0-1.0)
- `min_edges`: minimum number of edges (raises the probability on small networks)

### Edge Styling

- `opacity`: edge opacity (0.0 = invisible, 1.0 = opaque), the C++ `-opacity`
  (`edge_alpha` is a deprecated alias)
- `gradient_edges`: enable/disable gradient coloring

## Picture Size

The picture is exactly `width` x `height` pixels. The layout frame (1.6 x 1.2 times the
network radius, the C++ viewport) is scaled uniformly to fit and centered, so any aspect
ratio works without distorting the network; the legends sit in the margins of the frame.

## Component Circles

Optional component border circles:

- Drawn around connected components
- Useful for fragmented networks
- Controlled by `draw_circles` parameter

## Legends

Both legends are drawn in layout units at the positions of the C++ `generateNetworkFile`,
so they scale with the picture.

### Color Legend (Right Side)

One circle per index from the lowest (1 for k-cores and d-cores, 2 for k-denses) to the
maximum, in a column to the right of the network, each labeled in its own color. When
there are more than 15 indices only every `max // 15 + 1`-th one, counted from the top, is
labeled. For k-dense pictures the labels follow `measure`: `mcore` (default) prints the
m-core number (k-dense minus 2) under the title `m-core`; `kdense` prints the k-dense index.

### Degree Legend (Left Side)

Up to five sample nodes with degrees `dmax`, `dmax/4`, `dmax/16`, ... (down to 2), drawn
with the radius the nodes of that degree have in the picture, white on a black background
and gray on a white one. Weighted graphs show strengths `smax / 4^i` instead (unless no
strength exceeds 1, where the radii follow the degree law and so does the legend).

### Legend Configuration

- `show_color_legend`: Show/hide the shell/dense index color legend
- `show_degree_scale`: Show/hide the degree (node size) legend, as the C++ `-showDegreeScale`
  (`show_size_legend` is a deprecated alias)
- `legend_fontsize`: Manual font size in points (default: the C++ size, which scales with
  the picture)
- Text color: Automatic (white on dark, black on light)

## Background and Colors

- `background`: `black` (default) or `white`
- `color_scheme`: `col` (color), `bw` (black & white) or `bwi` (interlaced black & white)

**Recommendation**: Black background with color scheme for large networks (better contrast).

## Example Interpretation

### CAIDA AS-Relationships Visualization

Looking at a typical output:

**Center (red):**
- High k-core (k≈100-149 in the 2025 CAIDA snapshot)
- Tier-1 ISPs and backbone providers
- Dense interconnection
- Few nodes, many edges between them

**Middle rings (yellow/green/cyan):**
- Medium k-core (k=20-50)
- Regional ISPs and medium providers
- Moderate connectivity
- Transitional layer

**Outer rings (blue/magenta):**
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
config.visualization.opacity = 0.2        # Faint edges (the default)
config.visualization.node_size_scale = 0.4  # Smaller nodes
```

### For Small Networks (<1K nodes)

```python
config.visualization.edges_percent = 1.0  # Show all edges
config.visualization.opacity = 0.8        # More opaque
config.visualization.node_size_scale = 2.0  # Larger nodes
config.visualization.epsilon = 0.3        # Thicker rings
```

### For Publication-Quality

```python
config.visualization.width = 3600   # High resolution
config.visualization.height = 3600
config.visualization.background = "white"
config.visualization.opacity = 0.6
```

## Spiral Layout

`config.layout.use_spiral_layout` and the `spiral_*` settings are accepted but not
implemented (#18); the classic placement is always used.
