# LaNet-vi 5.0

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/lanet-vi)](https://pypi.org/project/lanet-vi/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/conexdat/LaNet-vi/workflows/CI/badge.svg)](https://github.com/conexdat/LaNet-vi/actions)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

**Large-scale network visualization using k-core decomposition**

LaNet-vi is a Python package for visualizing large-scale networks through hierarchical decomposition algorithms. It reveals network structure by identifying the k-core hierarchy - from peripheral nodes to densely connected cores.

## What is K-Core Decomposition?

K-core decomposition identifies hierarchical layers in networks where each k-core is a maximal subgraph with all nodes having at least k neighbors. This creates an "onion-like" structure revealing:

- **Core nodes** (high k): Densely connected, central, resilient
- **Peripheral nodes** (low k): Loosely connected, on the edges
- **Intermediate layers**: Transitional connectivity

Perfect for analyzing social networks, internet topology, biological networks, and collaboration graphs.

📖 **[Learn more about k-core concepts →](docs/concepts.md)**

## Features

- **K-core, k-dense, and d-core decomposition** algorithms
- **Circular hierarchical layout** with smooth rings and gradient edge coloring
- **High-performance rendering** for networks with millions of nodes
- **Flexible I/O** supporting compressed formats (gzip, bz2)
- **Community detection** with Louvain and modularity algorithms
- **Python API and CLI** with full configurability
- **Publication-ready** visualizations with auto-scaling legends

## Installation

```bash
# Using uv (recommended)
uv pip install lanet-vi

# Or with pip
pip install lanet-vi
```

## Quick Start

### Command Line

```bash
# Visualize a network
lanet-vi visualize --input network.txt --output viz.png

# With custom settings
lanet-vi visualize --input network.txt \
  --width 2400 --height 2400 \
  --background black \
  --output viz.png

# Generate configuration template
lanet-vi config my_config.yaml
```

### Python API

```python
import networkx as nx
from lanet_vi import Network, LaNetConfig, DecompositionType

# Load network
G = nx.karate_club_graph()

# Decompose and visualize
config = LaNetConfig()
net = Network(G, config)
net.decompose(DecompositionType.KCORES)
net.visualize("output.png")
```

## Example: Internet Topology

```python
from lanet_vi.io.readers import read_caida_snapshot
from lanet_vi import Network, LaNetConfig

# Download and visualize CAIDA AS-relationships data
graph, _ = read_caida_snapshot(
    "https://publicdata.caida.org/datasets/as-relationships/serial-1/20170101.as-rel.txt.bz2"
)

config = LaNetConfig()  # Uses optimized defaults
net = Network(graph, config)
net.decompose()
net.visualize("internet_topology.png")
```

**See example output:** [examples/outputs/caida_as_relationships_kcores.png](examples/outputs/caida_as_relationships_kcores.png)

The visualization reveals the Internet's hierarchical structure with Tier-1 providers in the center and stub networks at the periphery.

## Example Visualizations

<p align="center">
  <img src="examples/outputs/caida_as_relationships_kcores.png" width="45%" alt="K-cores decomposition">
  <img src="examples/outputs/caida_as_relationships_kdenses.png" width="45%" alt="K-denses decomposition">
  <br>
  <em>CAIDA AS-Relationships Network (56,345 nodes): K-cores (left) vs K-denses (right)</em>
</p>

The visualizations reveal the hierarchical structure of the Internet, with densely connected core networks (red/orange) at the center and peripheral networks (blue/purple) at the edges. K-cores use degree-based decomposition while k-denses use triangle-based decomposition, highlighting different structural properties.

## Input Format

Edge list (space or tab separated):

```
# Comments start with #
0 1
1 2
2 0
```

Weighted networks:

```
0 1 2.5
1 2 3.0
```

Supports `.txt`, `.txt.gz`, `.txt.bz2` formats.

## Common Options

**Decomposition:**
- `--decomp [kcores|kdenses|dcores]`: Decomposition algorithm (default: kcores)
- `--weighted`: Graph has edge weights
- `--directed`: Graph is directed (required for dcores)

**Visualization:**
- `--width`, `--height`: Image dimensions (default: 2400x2400)
- `--background [black|white]`: Background color (default: black)
- `--epsilon FLOAT`: Ring spread (default: 0.40)
- `--edges-percent FLOAT`: Percentage of edges to show (default: 0.5)
- `--edge-alpha FLOAT`: Edge transparency (default: 0.6)

**Output:**
- `--output PATH`: Visualization file (PNG, PDF, SVG)
- `--cores-file PATH`: Export decomposition data (CSV or JSON)

**Full CLI reference:** See [docs/usage.md](docs/usage.md#using-lanet-vi-via-command-line)

## Configuration

Generate a template:

```bash
lanet-vi config my_config.yaml
```

Example configuration:

```yaml
visualization:
  background: black
  width: 2400
  height: 2400
  epsilon: 0.40
  edges_percent: 0.5
  edge_alpha: 0.6

layout:
  seed: 0

decomposition:
  decomp_type: kcores
```

Use it:

```bash
lanet-vi visualize --input network.txt --config my_config.yaml
```

## Documentation

- **[K-Core Concepts](docs/concepts.md)** - Understanding k-core decomposition
- **[Visualization Guide](docs/visualization.md)** - How the plots work (colors, sizing, layout)
- **[Usage Guide](docs/usage.md)** - Detailed Python API and CLI examples
- **[Examples](examples/)** - Working examples with real datasets

## Advanced Features

### Community Detection

```bash
lanet-vi visualize --input network.txt \
  --detect-communities \
  --draw-community-boundaries \
  --output communities.png
```

### Random Graph Generation

```bash
lanet-vi generate --output test.txt \
  --model barabasi-albert \
  --nodes 1000 --edges 3
```

### D-Cores (Directed Networks)

```bash
lanet-vi visualize --input citations.txt \
  --directed --decomp dcores \
  --output dcores.png
```

## Performance Tips

**Large networks (>100K nodes):**

```python
config.visualization.edges_percent = 0.1  # Show 10% of edges
config.visualization.edge_alpha = 0.5
config.visualization.node_size_scale = 0.4
```

**Publication quality:**

```python
config.visualization.width = 3600
config.visualization.height = 3600
config.visualization.background = "white"
```

## Development

```bash
git clone https://github.com/conexdat/LaNet-vi.git
cd LaNet-vi
uv sync --all-extras
uv run pytest
```

## Citation

If you use LaNet-vi in your research, please cite:

- Alvarez-Hamelin, J.I., Dall'Asta, L., Barrat, A., Vespignani, A. (2006). "Large scale networks fingerprinting and visualization using the k-core decomposition". *Advances in Neural Information Processing Systems 18*.

- Beiró, M.G., Alvarez-Hamelin, J.I., Busch, J.R. (2008). "A low complexity visualization tool that helps to perform complex systems analysis". *New Journal of Physics*.

## License

MIT License

## Authors

- Esteban Carisimo (Python implementation)
- Mariano Beiró (original C++ version)
- J. Ignacio Alvarez-Hamelin (original C++ version)

