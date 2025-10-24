# LaNet-vi Quick Start Guide

## Installation

```bash
cd /Users/estcarisimo/postdoc/public-coding/LaNet-vi
uv sync
```

## Running with CAIDA Data (20251001.as-rel.txt.bz2)

### Option 1: Python Script (Easiest)

```bash
uv run python examples/caida_example.py
```

**What happens:**
- Downloads CAIDA snapshot automatically
- Parses ~75,000 AS nodes
- Computes k-core decomposition
- Generates: `caida_as_relationships_kcores.png`
- Exports: `caida_cores.csv`

### Option 2: CLI (Manual Control)

```bash
# Download data first
wget https://publicdata.caida.org/datasets/as-relationships/serial-1/20251001.as-rel.txt.bz2

# Visualize
uv run lanet-vi visualize \
    -input 20251001.as-rel.txt.bz2 \
    -output caida_viz.png \
    -decomp kcores \
    -W 1920 -H 1080 \
    -edges 0.02 \
    -coresfile cores.csv
```

### Option 3: Shell Script

```bash
./examples/caida_cli_example.sh
```

Generates 3 visualizations:
- High-res k-core (white background)
- Dark theme k-core
- K-dense decomposition

## Quick Test (No Download)

```bash
uv run python examples/basic_usage.py
```

Uses NetworkX's built-in Karate Club graph.

## Command Reference

### Get Network Info

```bash
uv run lanet-vi info network.txt
```

### Basic Visualization

```bash
uv run lanet-vi visualize -input network.txt -output viz.png
```

### K-dense Decomposition

```bash
uv run lanet-vi visualize \
    -input network.txt \
    -decomp kdenses \
    -output kdense.png
```

### High-Resolution + Dark Theme

```bash
uv run lanet-vi visualize \
    -input network.txt \
    -W 1920 -H 1080 \
    -bckgnd black \
    -output hd_dark.png
```

### Weighted Network

```bash
uv run lanet-vi visualize \
    -input weighted.txt \
    -weighted \
    -granularity 10 \
    -output weighted_viz.png
```

### Export Decomposition Only

```bash
uv run lanet-vi visualize \
    -input network.txt \
    -coresfile cores.csv \
    -nographic
```

## Python API

### Basic Usage

```python
from lanet_vi import Network, DecompositionType

# Load from file
net = Network.from_edge_list("network.txt")

# Or use NetworkX graph
import networkx as nx
G = nx.karate_club_graph()
net = Network(G)

# Decompose and visualize
net.decompose(DecompositionType.KCORES)
net.visualize("output.png")
```

### With Configuration

```python
from lanet_vi import Network, LaNetConfig, DecompositionType

config = LaNetConfig()
config.visualization.width = 1920
config.visualization.height = 1080
config.visualization.background = "black"
config.layout.seed = 42

net = Network.from_edge_list("network.txt", config)
net.decompose(DecompositionType.KCORES)
net.visualize("output.png")
```

### CAIDA Data

```python
from lanet_vi import Network
from lanet_vi.io.readers import read_caida_snapshot

# Download and parse
url = "https://publicdata.caida.org/.../20251001.as-rel.txt.bz2"
graph, dataframe = read_caida_snapshot(url)

# Visualize
net = Network(graph)
net.decompose()
net.visualize("caida.png")
```

## Input Format

### Edge List (space-separated)

```
0 1
1 2
2 0
```

### Weighted Edges

```
0 1 2.5
1 2 3.0
2 0 1.5
```

### CAIDA Format (pipe-separated)

```
# provider|customer|relationship_type
1|2|-1
3|4|0
```

## Common Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `-input` | Input edge list file | `network.txt` |
| `-output` | Output image file | `viz.png` |
| `-decomp` | kcores or kdenses | `kcores` |
| `-weighted` | Enable edge weights | flag |
| `-W` / `-H` | Image dimensions | `1920` / `1080` |
| `-bckgnd` | white or black | `black` |
| `-color` | col, bw, or bwi | `col` |
| `-edges` | % of visible edges | `0.02` |
| `-minedges` | Min number of edges | `5000` |
| `-coresfile` | Export decomposition | `cores.csv` |
| `-seed` | Random seed | `42` |

## Troubleshooting

### "Module not found"

```bash
# Re-sync dependencies
uv sync
```

### "Command not found: lanet-vi"

```bash
# Use uv run prefix
uv run lanet-vi --help
```

### Large networks are slow

```bash
# Reduce edge visibility
uv run lanet-vi visualize \
    -input large.txt \
    -edges 0.01 \
    -minedges 1000
```

## Help

```bash
uv run lanet-vi --help
uv run lanet-vi visualize --help
uv run lanet-vi info --help
```

## Documentation

- `README.md` - Full documentation
- `examples/README.md` - More examples
- `REFACTOR_SUMMARY.md` - Technical details
