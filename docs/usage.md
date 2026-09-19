# Usage Guide

LaNet-vi can be used from the command line (`lanet-vi`) or as a Python library
(`lanet_vi`). Both run the same pipeline: load a network → decompose it (k-cores, k-denses
or d-cores) → compute the layout → render. This guide is the reference for both; the
concepts are in [concepts.md](concepts.md) and the picture itself is explained in
[visualization.md](visualization.md). Users of the C++ LaNet-vi 3.x will find the flag
translation in [cpp-migration.md](cpp-migration.md).

## Command line

### Commands

| Command | What it does |
|---|---|
| `lanet-vi visualize --input FILE --output FILE [options]` | Decompose and draw a network |
| `lanet-vi info FILE [--weighted]` | Print node/edge counts, degree statistics and the decomposition range |
| `lanet-vi config FILE [--decomp kcores\|kdenses\|dcores]` | Write a YAML configuration template with every setting and its default |
| `lanet-vi generate --output FILE --nodes N [--model ...]` | Write a random graph as an edge list |

### Quick start

```bash
lanet-vi visualize --input network.txt --output network.png
```

```bash
# The network statistics first
lanet-vi info network.txt
```

```bash
# K-dense (m-core) instead of k-cores, and the decomposition as CSV next to the picture
lanet-vi visualize --input network.txt --output kdense.png --decomp kdenses --cores-file kdense.csv
```

```bash
# Directed graph, d-cores
lanet-vi visualize --input citations.txt --output dcores.png --directed --decomp dcores
```

```bash
# Weighted graph: strength-based cores, 20 intervals with the same number of nodes each
lanet-vi visualize --input weighted.txt --output weighted.png \
  --weighted --granularity 20 --strength-intervals equalNodesPerInterval
```

```bash
# A large network: draw 10 % of the edges (at least 50,000), white background, wide picture
lanet-vi visualize --input big.txt --output big.png \
  --edges-percent 0.1 --min-edges 50000 --background white --width 3200 --height 2400
```

```bash
# The central half of the picture at full pixel size: a 2x zoom on the core
# (give --width/--height the window's aspect ratio, 4:3 here, for an exact crop)
lanet-vi visualize --input network.txt --output core.png \
  --window 0.25 0.75 0.25 0.75 --width 2400 --height 1800
```

```bash
# Reproducible pictures: the seed fixes the random choices of the layout and the edge sample
lanet-vi visualize --input network.txt --output network.png --seed 42
```

```bash
# A synthetic network to try things on
lanet-vi generate --output ba.txt --model barabasi-albert --nodes 2000 --edges 3 --seed 1
```

### `visualize` options

The C++ column gives the flag of LaNet-vi 3.x with the same meaning (see
[cpp-migration.md](cpp-migration.md)). Options marked *inert* are accepted for
compatibility and do nothing yet; their `--help` text names the tracking issue.

**Input and output**

| Option | Default | Meaning | C++ |
|---|---|---|---|
| `--input`, `-i PATH` | required | Edge list (see [Input format](#input-format)); `.gz` / `.bz2` are read transparently | `-input` |
| `--output`, `-o PATH` | `output.png` | Picture; the extension selects PNG, PDF or SVG | `-output` |
| `--config`, `-c PATH` | | YAML settings file (see [Configuration file](#configuration-file)) | |
| `--cores-file PATH` | | Write the decomposition: `.json` for the full result, anything else as CSV `node,index` | `-coresfile` |
| `--names PATH` | | Node names, one `node name` per line; turns `--node-labels` on | `-names` |
| `--colors-file PATH` | | Node colors, `node r g b` per line (values in 0–1); nodes absent from the file are drawn white on black / black on white and the color legend is hidden | `-colorsFile` |
| `--weighted`, `-w` | off | Third column is an edge weight; cores are computed on strengths | `-weighted` |
| `--multigraph` | off | Keep repeated edges (they count in the degree; with `--weighted` their weights add up) | `-multigraph` |
| `--directed` | off | Read a directed graph (required for `--decomp dcores`) | `-directed` (4.0) |

**Decomposition**

| Option | Default | Meaning | C++ |
|---|---|---|---|
| `--decomp`, `-d kcores\|kdenses\|dcores` | `kcores` | Which decomposition | `-decomp` |
| `--measure mcore\|kdense` | `mcore` | Numbering of the k-dense legend: `mcore` labels each k-dense as k − 2 (the m-core) and reads `--color-scale-max` in those units; `kdense` keeps k | `-measure` |
| `--no-cliques` | off | Spread the top core uniformly instead of by cliques | `-nocliques` |
| `--from-layer K` | 0 | *inert* (#23): induced subgraph of index ≥ K | `-fromlayer` |
| `--granularity N` | max degree | Weighted graphs: number of strength intervals | `-granularity` |
| `--strength-intervals equalIntervalSize\|equalNodesPerInterval\|equalLogIntervalSize\|custom` | `equalIntervalSize` | Weighted graphs: how the intervals are built ([concepts.md](concepts.md#weighted-k-cores)) | `-strengthsIntervals` |
| `--maximum-strength S` | data | Weighted graphs: top of the strength scale, to compare pictures of different networks | `-maximumStrength` |
| `--strength-intervals-file PATH` | | Interval boundaries, one per line, for `--strength-intervals custom` | `-strengthsIntervalsFile` (4.0) |

**Layout**

| Option | Default | Meaning | C++ |
|---|---|---|---|
| `--epsilon E` | 0.18 | Ring thickness as a fraction of its radius (formula (1) of the paper) | `-eps` |
| `--delta D` | 1.3 | Shrink factor of sibling components (`classic`) | `-delta` |
| `--gamma G` | 1.5 | Component diameter; scales the whole picture | `-gamma` |
| `--coord-distribution classic\|pow\|log` | `classic` | Placement of sibling components: concentric rings, or circle packing ([visualization.md](visualization.md#the-pow-and-log-coordinate-distributions)) | `-coordDistributionAlgorithm` |
| `--alpha A`, `--beta B` | 0.3, 1.0 | Constant and exponent of the disc area law of the packing (`pow` / `log`) | `-alpha`, `-beta` |
| `--ratio-constant C` | auto | Node radius factor of `pow` / `log` | `-ratioConstant` |
| `--seed N` | 0 | Random seed: cluster order, ties, angle frames, the packing and the edge sample | `-seed` |
| `--use-spiral-layout`, `--spiral-K`, `--spiral-beta`, `--spiral-separation` | off | *inert* (#18) | |

**Picture**

| Option | Default | Meaning | C++ |
|---|---|---|---|
| `--width`, `-W PX`; `--height`, `-H PX` | 2400 × 2400 | Picture size in pixels; any aspect ratio (the layout is scaled uniformly, legends in the margins) | `-W`, `-H` |
| `--window HSTART HEND VSTART VEND` | 0 1 0 1 | Render only that part of the picture, as fractions from the top-left corner, at the full pixel size | `-window` |
| `--background black\|white` | `black` | Background | `-bckgnd` |
| `--color-scheme col\|bw\|bwi` | `col` | Rainbow (red for the top core), grayscale, or interlaced grayscale | `-color` |
| `--color-scale-max K` | max index | Indices at or above K share the top color (m-core units for k-dense with `--measure mcore`) | `-colorScaleMaxValue` |
| `--edges-percent P` | 0.5 | Fraction of edges drawn; each edge is kept with probability `max(P, min_edges / E)` | `-edges` |
| `--min-edges N` | 50000 | Lower bound on the number of edges drawn | `-minedges` |
| `--opacity O` | 0.2 | Edge opacity (`--edge-alpha` is a deprecated alias) | `-opacity` |
| `--gradient-edges` / `--no-gradient-edges` | on | Each half of an edge takes the color of the opposite endpoint | |
| `--node-size-scale S` | 1.0 | Multiplier on the node radius (1.0 is the C++ size) | |
| `--node-edge-color COLOR` | none | Border color of the nodes (the C++ drew none) | |
| `--draw-circles` | off | Draw the disc of every component | `-drawCircles` |
| `--show-color-legend` / `--no-show-color-legend` | on | Index color legend (right margin) | |
| `--show-degree-scale` / `--no-show-degree-scale` | on | Degree (node size) legend (left margin); `--show-size-legend` is a deprecated alias | `-showDegreeScale` |
| `--node-labels` / `--no-node-labels` | on with `--names` | Draw node names | |
| `--font-zoom Z` | 1.0 | Font size multiplier for node names | `-font` |
| `--legend-fontsize PT` | auto | Legend font size in points (default: the C++ size, relative to the picture) | |
| `--min-edge-width`, `--max-edge-width` | | *inert*, deprecated: widths follow the C++ rule | |

**Communities** — `--detect-communities`, `--community-algorithm`, `--community-resolution`,
`--color-by-community`, `--draw-community-boundaries`: *inert* (#23); community detection is
available in the Python API (`lanet_vi.community`) but not wired into the CLI rendering.

**Logging** — `--verbose`, `-v` (debug output), `--quiet`, `-q` (no console output),
`--log-file PATH` (also write the log to a file). `lanet-vi generate` and `lanet-vi info`
take the same three.

### `generate` options

`--model erdos-renyi|barabasi-albert|watts-strogatz|powerlaw-cluster` (default
`erdos-renyi`), `--nodes N` (required), and per model: `--probability P` or `--edges M`
(Erdős–Rényi G(n, p) / G(n, m)), `--edges M` (Barabási–Albert attachment, powerlaw-cluster),
`--neighbors K --rewire P` (Watts–Strogatz), `--triangle-prob P` (powerlaw-cluster);
`--directed` (Erdős–Rényi only; the other models ignore it), `--weighted` (random
weights), `--seed N`. The output is a plain edge list
that `visualize` reads back.

### Configuration file

`lanet-vi config settings.yaml` writes every setting with its default, grouped as
`graph`, `decomposition`, `visualization`, `layout` and `community`; edit it and pass it
with `--config`. Precedence is that of the C++: built-in defaults < the YAML file < flags
given explicitly on the command line, so a file can hold the standing choices and a flag
override one of them:

```bash
lanet-vi config settings.yaml
lanet-vi visualize --input network.txt --output network.png --config settings.yaml --background white
```

The same file loads in Python with `lanet_vi.load_config_from_yaml(path)` and can be
written with `save_config_to_yaml(config, path)`. A minimal file:

```yaml
decomposition:
  decomp_type: kcores
visualization:
  background: black
  width: 2400
  height: 2400
  edges_percent: 0.5
  opacity: 0.2
layout:
  seed: 0
```

### Input format

An edge per line, `source target [weight]`, separated by spaces or tabs; lines starting
with `#` are comments; `.gz` and `.bz2` files are decompressed on the fly. Node ids can
be integers or names (names with spaces need a `--names` file). Self-loops are dropped;
repeated edges are merged unless `--multigraph`. Without `--weighted` a third column is
ignored; with it, a missing weight counts as 1.

```
# source target weight
0 1 2.5
1 2 3.0
2 0
```

CAIDA AS-relationship snapshots (`<as1>|<as2>|<relation>` lines) are read by
`lanet_vi.io.readers.read_caida_snapshot`, from a local file or a URL.

## Python API

```python
import networkx as nx
from lanet_vi import DecompositionType, LaNetConfig, Network

# networkx's karate club carries edge weights, which LaNet-vi would detect and use
# (strength-based cores); nx.Graph(G.edges()) keeps the plain graph
G = nx.Graph(nx.karate_club_graph().edges())
net = Network(G, LaNetConfig())
result = net.decompose(DecompositionType.KCORES)
net.visualize("karate.png")

print(result.min_index, result.max_index)   # 1 4: the k-core range
print(result.node_indices[0])               # 4: the k-core number of node 0
```

Edge weights are detected automatically: a graph whose edges carry a `weight` attribute
(or a third column with `--weighted`) gets strength-based cores; see
[concepts.md](concepts.md#weighted-k-cores).

`Network` takes a NetworkX graph (`Graph`, `DiGraph`, `MultiGraph`) or reads an edge list:

```python
net = Network.from_edge_list("network.txt", LaNetConfig())
```

Every setting of the CLI is a field of `LaNetConfig`, in the same groups as the YAML file
(the CLI flag `--edges-percent` is `config.visualization.edges_percent`, and so on):

```python
from lanet_vi import DecompositionType, LaNetConfig
from lanet_vi.models.config import BackgroundColor, CoordDistributionAlgorithm

config = LaNetConfig()
config.decomposition.decomp_type = DecompositionType.KDENSES
config.visualization.width = 3200
config.visualization.height = 2400
config.visualization.background = BackgroundColor.WHITE
config.visualization.edges_percent = 0.1
config.layout.coord_distribution = CoordDistributionAlgorithm.POWER
config.layout.seed = 42
```

The pipeline can be run step by step when the intermediate results are needed:

```python
result = net.decompose()          # DecompositionResult
layout = net.compute_layout()     # VisualizationLayout: positions, colors, sizes, edges
net.visualize("network.png", layout)
```

- `result.node_indices` maps every node to its index (k-core number, k-dense index, or
  `max(k_in, k_out)` for d-cores; the pairs are in `result.metadata["d_cores"]`);
  `result.components` lists the connected pieces of each shell; k-dense results also carry
  `result.metadata["edge_indices"]`.
- `layout.node_positions`, `layout.node_colors` (RGB in 0–1), `layout.node_sizes` (layout
  units), `layout.visible_edges`, `layout.components` (nested components with center and
  radius) and `layout.bounds` are what the renderer draws.
- `net.get_metadata()` returns node and edge counts, density and degree statistics.

Results can be written with the same writers the CLI uses:

```python
from lanet_vi.io.writers import write_decomposition_csv, write_decomposition_json

write_decomposition_csv(result, "cores.csv")    # node,index
write_decomposition_json(result, "cores.json")  # indices, components, metadata
```

### Community detection (API only)

`lanet_vi.community` wraps NetworkX's Louvain and greedy-modularity algorithms and
`lanet_vi.metrics` provides partition comparison metrics (NMI and friends). They are
NetworkX-based replacements, not ports of the C++ code, and they are not connected to
the CLI or the renderer yet (#23, #26).

## Notes on large networks

- The CAIDA AS graph (78k nodes, 489k edges) takes about a minute end to end at the
  default 2400 × 2400; the layout is the expensive part and grows with the number of
  nodes and shells.
- Drawing is dominated by the edges: `--edges-percent` (with `--min-edges` as a floor)
  is the knob. On a black background 0.1–0.5 with the default opacity reads well; on
  white, raise the opacity.
- `--seed` makes the picture reproducible; a different seed only changes the random
  choices (cluster order, angle frames, the edge sample), not the structure.
- Everything is in memory; a few million edges are fine on a laptop, but the
  matplotlib rendering of millions of edges is slow — sample them.
