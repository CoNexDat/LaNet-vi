# Usage Guide

## Using LaNet-vi as a Python Library

### Basic Example

```python
import networkx as nx
from lanet_vi import Network, LaNetConfig, DecompositionType

# Create or load a network
G = nx.karate_club_graph()

# Use default configuration
config = LaNetConfig()
net = Network(G, config)

# Decompose and visualize
result = net.decompose(DecompositionType.KCORES)
net.visualize("karate_club_kcores.png")

# Access decomposition results
print(f"K-core range: {result.min_index} - {result.max_index}")
print(f"Number of components: {len(result.components)}")
```

### Loading from Edge List

```python
from lanet_vi import Network, LaNetConfig

# Load from file
config = LaNetConfig()
net = Network.from_edge_list("network.txt", config)

# Decompose and visualize
net.decompose()
net.visualize("output.png")
```

### Custom Configuration

```python
from lanet_vi import LaNetConfig

# Create custom configuration
config = LaNetConfig()

# Visualization settings
config.visualization.width = 2400
config.visualization.height = 2400
config.visualization.background = "black"
config.visualization.epsilon = 0.40

# Edge settings
config.visualization.edges_percent = 0.5
config.visualization.edge_alpha = 0.6

# Layout settings
config.layout.seed = 0

# Use with network
net = Network(G, config)
net.decompose()
net.visualize("custom_viz.png")
```

### Accessing Decomposition Data

```python
# Run decomposition
result = net.decompose(DecompositionType.KCORES)

# Node k-core values
for node, k_core in result.node_indices.items():
    print(f"Node {node}: k-core = {k_core}")

# Components
for i, component in enumerate(result.components):
    print(f"Component {i}: {len(component.nodes)} nodes")
    print(f"  Shell index: {component.shell_index}")
    print(f"  Nodes: {component.nodes[:5]}...")  # First 5 nodes

# Metadata
metadata = net.get_metadata()
print(f"Nodes: {metadata['num_nodes']}")
print(f"Edges: {metadata['num_edges']}")
print(f"Average degree: {metadata['avg_degree']:.2f}")
```

### Advanced: Loading from Custom Source

```python
import networkx as nx
from lanet_vi import Network, LaNetConfig

# Create network from any source
edges = [(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)]
G = nx.Graph(edges)

# Add node attributes
for node in G.nodes():
    G.nodes[node]['label'] = f"Node_{node}"

# Visualize
config = LaNetConfig()
net = Network(G, config)
net.decompose()
net.visualize("custom_network.png")
```

### Exporting Results

```python
from lanet_vi.io.writers import write_decomposition_csv, write_decomposition_json

# Decompose
result = net.decompose()

# Export as CSV
write_decomposition_csv(result, "kcores.csv")

# Export as JSON (full details)
write_decomposition_json(result, "kcores.json")
```

## Using LaNet-vi via Command Line

### Basic Usage

```bash
# Visualize a network
lanet-vi visualize --input network.txt --output viz.png

# Show network info
lanet-vi info network.txt

# Generate config template
lanet-vi config my_config.yaml
```

### Common Workflows

**Analyze unknown network:**

```bash
# First, check network statistics
lanet-vi info network.txt

# Then visualize with defaults
lanet-vi visualize --input network.txt --output viz.png

# Export decomposition data
lanet-vi visualize --input network.txt \\
  --cores-file kcores.csv \\
  --output viz.png
```

**Customize visualization:**

```bash
# Large network with custom settings
lanet-vi visualize --input large_network.txt \\
  --width 2400 --height 2400 \\
  --background black \\
  --edges-percent 0.1 \\
  --edge-alpha 0.5 \\
  --node-size-scale 0.4 \\
  --output large_viz.png
```

**Weighted network:**

```bash
lanet-vi visualize --input weighted_network.txt \\
  --weighted \\
  --decomp kdenses \\
  --output kdense_viz.png
```

**Using configuration file:**

```bash
# Generate template
lanet-vi config viz_config.yaml

# Edit viz_config.yaml with your settings

# Use it
lanet-vi visualize --input network.txt \\
  --config viz_config.yaml \\
  --output viz.png
```

### Configuration File Example

Generate with `lanet-vi config template.yaml`, then edit:

```yaml
visualization:
  background: black
  width: 2400
  height: 2400
  epsilon: 0.40
  edges_percent: 0.5
  edge_alpha: 0.6
  min_edges: 50000
  node_size_scale: 0.5

layout:
  seed: 0
  min_component_size: 10

decomposition:
  decomp_type: kcores
  no_cliques: false
```

Use it:

```bash
lanet-vi visualize --input network.txt --config template.yaml
```

**Override config values via CLI:**

```bash
# Config sets defaults, CLI flags override
lanet-vi visualize --input network.txt \\
  --config template.yaml \\
  --width 3600 \\
  --background white
```

### Debugging and Logging

```bash
# Verbose output
lanet-vi visualize --input network.txt --verbose

# Log to file
lanet-vi visualize --input network.txt \\
  --log-file processing.log \\
  --verbose

# Quiet mode (only file logging)
lanet-vi visualize --input network.txt \\
  --quiet \\
  --log-file processing.log
```

## Integration Examples

### Jupyter Notebook

```python
from IPython.display import Image, display
from lanet_vi import Network, LaNetConfig
import networkx as nx

# Create and visualize
G = nx.karate_club_graph()
config = LaNetConfig()
net = Network(G, config)
net.decompose()
net.visualize("karate.png")

# Display inline
display(Image("karate.png"))
```

### Pipeline with NetworkX

```python
import networkx as nx
from lanet_vi import Network, LaNetConfig

# Load and process with NetworkX
G = nx.read_edgelist("network.txt")

# NetworkX analysis
print(f"Clustering coefficient: {nx.average_clustering(G):.3f}")
print(f"Average path length: {nx.average_shortest_path_length(G):.2f}")

# Visualize with LaNet-vi
config = LaNetConfig()
net = Network(G, config)
net.decompose()
net.visualize("network_viz.png")
```

### Batch Processing

```python
from pathlib import Path
from lanet_vi import Network, LaNetConfig

# Process multiple networks
networks = Path("data/").glob("*.txt")
config = LaNetConfig()

for network_file in networks:
    print(f"Processing {network_file.name}...")

    net = Network.from_edge_list(str(network_file), config)
    result = net.decompose()

    output = f"output/{network_file.stem}_kcores.png"
    net.visualize(output)

    print(f"  K-core range: {result.min_index}-{result.max_index}")
    print(f"  Saved: {output}")
```

## Best Practices

### Performance

- **Large networks (>100K nodes):** Use `edges_percent < 0.2` to reduce rendering time
- **Memory:** Process very large networks in chunks or use `min_component_size` to filter small components
- **Speed:** Use `use_spatial_hashing=True` (default) for fast layout

### Visualization Quality

- **Resolution:** Use 2400x2400 or higher for publication
- **Background:** Black background works better for large networks (better contrast)
- **Epsilon:** Adjust based on network size (0.2-0.4 range for most networks)
- **Seed:** Use `seed=0` for maximum uniformity, or specific seeds for reproducibility

### Debugging

- Start with `lanet-vi info` to understand network structure
- Use `--verbose` to see detailed processing steps
- Check `--log-file` output for error diagnostics
- Visualize small sample first before processing large dataset
