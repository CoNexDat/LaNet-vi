# Visualization Guide

## How LaNet-vi Plots Work

LaNet-vi creates hierarchical network visualizations using a circular, shell-based layout that reveals the k-core structure.

## Layout Algorithm

### Circular Shell Layout

Nodes are positioned in concentric rings based on their k-core number:

```
Higher k-core → Closer to center → Inner rings
Lower k-core → Further from center → Outer rings
```

**Algorithm:**
1. Each k-shell gets a radius: `r = (max_k - k + 1) * spacing`
2. Nodes are placed around the ring at that radius
3. Angular position determined by neighbor relationships
4. Radial jitter (`epsilon`) adds variation within each ring

**Parameters:**
- `epsilon`: Controls radial spread (0.0 = tight line, 1.0 = wide band)
- `seed`: Random seed for reproducible layouts

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

- `show_degree_scale`: Show/hide k-core legend
- `show_size_legend`: Show/hide degree legend
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
- High k-core (k=60-79)
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
config.visualization.node_size_scale = 1.0  # Normal size
config.visualization.epsilon = 0.3        # More spread
```

### For Publication-Quality

```python
config.visualization.width = 3600   # High resolution
config.visualization.height = 3600
config.visualization.background = "white"
config.visualization.edge_alpha = 0.6
```

## Advanced: Spiral Layout

Alternative to circular layout:

```python
config.layout.use_spiral_layout = True
config.layout.spiral_K = 15.0
config.layout.spiral_beta = 2.0
```

Creates curved, semicircular arrangements useful for aesthetic presentations.
