# LaNet-vi Examples

This directory contains examples demonstrating LaNet-vi usage.

## CAIDA AS-Relationships Example

**File:** `caida_example.py`

Visualizes the CAIDA AS-Relationships dataset showing Internet topology at the Autonomous System (AS) level.

### Running the Example

```bash
uv run python examples/caida_example.py
```

### What it Does

1. **Downloads data** from CAIDA public dataset (20170101 snapshot)
2. **Loads network** using pandas-based edge list reader
3. **Computes k-core decomposition** using NetworkX
4. **Generates visualization** with optimized default settings
5. **Exports results** to CSV file

### Network Statistics

- **Nodes:** 56,345 Autonomous Systems
- **Edges:** 239,876 AS relationships
- **K-core range:** 1 to 79
- **Components:** 50,870

### Output Files

- `caida_as_relationships_kcores.png` - Visualization (see `outputs/`)
- `caida_cores.csv` - K-core values for each AS

### Interpretation

The visualization reveals the hierarchical structure of the Internet:

**Inner core (red/orange, k=60-79):**
- Tier-1 backbone providers (Level3, Telia)
- Hypergiant networks (Google, AWS, Cloudflare, Microsoft)
- Dense interconnection
- Critical infrastructure

**Middle rings (yellow/green, k=20-60):**
- Regional and national ISPs
- Medium-sized content providers
- Transit providers
- Moderate connectivity

**Outer rings (blue/purple, k=1-20):**
- Stub networks and end-user networks
- Small ISPs and organizations
- Edge of the network
- Sparse connections

### Customization

The example uses optimized default settings. To customize:

```python
config = LaNetConfig()

# Adjust visualization
config.visualization.width = 3600      # Higher resolution
config.visualization.epsilon = 0.50    # More radial spread
config.visualization.edges_percent = 0.3  # More edges

# Adjust layout
config.layout.seed = 42  # Different random seed
```

### Data Source

Data is downloaded from: https://publicdata.caida.org/datasets/as-relationships/

The CAIDA AS Relationships Dataset provides inferred relationships between Autonomous Systems based on BGP routing data.

### Example Output

![CAIDA AS-Relationships K-Core Visualization](outputs/caida_as_relationships_kcores.png)

The visualization clearly shows the onion-like structure of the Internet, with a dense core of highly interconnected backbone providers and an enormous periphery of loosely connected stub networks.

## CAIDA AS-Relationships K-Denses Example

**File:** `caida_kdenses_example.py`

Visualizes the same CAIDA AS-Relationships dataset using k-denses (m-core) decomposition, which reveals tightly-knit communities based on triangle density rather than degree.

### Running the Example

```bash
uv run python examples/caida_kdenses_example.py
```

### What it Does

1. **Downloads data** from CAIDA public dataset (20170101 snapshot)
2. **Loads network** using pandas-based edge list reader
3. **Computes k-denses decomposition** using triangle counting
4. **Generates visualization** with optimized default settings
5. **Exports results** to CSV file

### Network Statistics

- **Nodes:** 56,345 Autonomous Systems
- **Edges:** 239,876 AS relationships
- **K-dense range:** 2 to 56
- **Components:** 48,206

### Output Files

- `caida_as_relationships_kdenses.png` - Visualization (see `outputs/`)
- `caida_denses.csv` - K-dense values for each AS

### K-Denses vs K-Cores

**K-cores (degree-based):**
- Identifies hierarchical structure
- Node in k-core if it has ≥k neighbors
- Wide range of shells (1-79 for this dataset)
- Shows overall connectivity patterns

**K-denses (triangle-based):**
- Identifies cohesive communities
- Node in k-dense if it participates in ≥k triangles
- Narrower range of shells (2-56 for this dataset)
- Reveals tightly-knit groups with strong interconnections

### Interpretation

The k-denses visualization reveals cohesive communities in Internet topology:

**Inner core (red/orange, k=40-56):**
- Most densely interconnected AS groups
- Hypergiants and Tier-1 providers forming tight meshes
- High triangle density (many mutual peering relationships)
- Critical infrastructure with redundant paths

**Middle rings (yellow/green, k=10-40):**
- Regional ISP communities
- Transit provider clusters
- Moderate triangle density
- Local interconnection hubs

**Outer rings (blue/purple, k=2-10):**
- Peripheral AS groups
- Stub networks with minimal peering
- Low triangle density
- Edge communities

### Comparison with K-Cores

K-denses produces fewer shells than k-cores for the same network (56 vs 79), indicating that triangle-based decomposition is more selective. While k-cores shows the broad hierarchical structure, k-denses highlights where ASes form tightly-knit communities with mutual relationships.

For Internet topology analysis:
- Use **k-cores** to understand overall hierarchical structure
- Use **k-denses** to identify cohesive peering communities

### Customization

The example uses optimized default settings. To customize:

```python
config = LaNetConfig()

# Adjust visualization
config.visualization.width = 3600      # Higher resolution
config.visualization.epsilon = 0.50    # More radial spread
config.visualization.edges_percent = 0.3  # More edges

# Adjust layout
config.layout.seed = 42  # Different random seed
```

### Data Source

Data is downloaded from: https://publicdata.caida.org/datasets/as-relationships/

The CAIDA AS Relationships Dataset provides inferred relationships between Autonomous Systems based on BGP routing data.

### Example Output

![CAIDA AS-Relationships K-Denses Visualization](outputs/caida_as_relationships_kdenses.png)

The visualization reveals cohesive communities in the Internet's AS topology, with a concentrated core of ASes that participate in many mutual peering triangles and a periphery of loosely connected stub networks.
