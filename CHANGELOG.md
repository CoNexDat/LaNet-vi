# Changelog

All notable changes to LaNet-vi will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [5.0.0] - 2025-10-18

### Overview

Complete Python refactor of LaNet-vi 3.x (C++) with all legacy features included. This version brings all functionality from the latest C++ codebase (previously in `legacy/Source/`) into a modern Python implementation using NetworkX, pandas, and matplotlib.

### Core Features (Ported from C++ LaNet-vi 3.x)

- **K-core decomposition**: Classic k-core algorithm using NetworkX
- **K-dense decomposition**: Triangle-based decomposition (m-cores)
- **D-core decomposition**: Directed graphs with (k_in, k_out) pairs per node
  - Ported from `legacy/Source/graph_dcores.cpp`
  - CLI: `--directed --decomp dcores`
- **Spiral/semicircular layout**: Mathematical spiral placement using Newton-Raphson solver
  - Ported from `legacy/Source/espiral.cpp`
  - CLI: `--use-spiral-layout`
- **Community detection**: Louvain and greedy modularity algorithms
  - Ported from `legacy/Source/community.cpp`
  - CLI: `--detect-communities`
- **Community visualization**: Color-coded nodes with boundary overlays
- **Random graph generation**: Testing and benchmarking utilities
  - Erdős-Rényi, Barabási-Albert, Watts-Strogatz, Powerlaw cluster
  - Ported from `legacy/Source/erdos_renyi.cpp`
  - CLI: `lanet-vi generate`

### New Python-Specific Features

- **Type-safe configuration**: Pydantic models for all settings
- **Modern CLI**: Typer + Rich with `--double-hyphen` flags (Unix/GNU standard)
- **Comprehensive logging**: DEBUG and INFO levels (`--verbose`, `--quiet`, `--log-file`)
- **Enhanced JSON export**: D3.js-compatible graph exports with full metadata
- **Spatial indexing**: KD-tree based indexing for O(log N) queries on large graphs
- **Information theory metrics**: MI, NMI, ARI, VI for partition comparison
- **Pandas integration**: Efficient data management for large networks
- **YAML configuration**: Easy-to-edit config files

### Architecture Improvements

- **NetworkX integration**: Leverages battle-tested graph algorithms
- **Modular design**: Clean separation of concerns (io, decomposition, visualization, metrics)
- **Better performance**: NumPy vectorization, spatial indexing, optimized layouts
- **No external renderers**: Pure Python/Matplotlib (removed POV-Ray dependency)
- **Comprehensive testing**: Type checking with mypy, linting with ruff

### Dependencies

Core libraries:
- `networkx>=3.0` - Graph algorithms
- `pandas>=2.0` - Data management
- `matplotlib>=3.7` - Visualization
- `scipy>=1.10` - Spatial indexing, convex hulls
- `scikit-learn>=1.3` - Clustering metrics
- `pydantic>=2.0` - Configuration validation
- `typer>=0.9` + `rich>=13.0` - CLI interface

### Usage Examples

```bash
# Basic k-core visualization
lanet-vi visualize --input network.txt --output viz.png

# D-core decomposition on directed networks
lanet-vi visualize --input network.txt --directed --decomp dcores --output dcores.png

# Community detection
lanet-vi visualize --input network.txt --detect-communities --output communities.png

# Spiral layout
lanet-vi visualize --input network.txt --use-spiral-layout --output spiral.png

# Generate random graph for testing
lanet-vi generate --output test.txt --model barabasi-albert --nodes 1000 --edges 3

# Verbose logging
lanet-vi visualize --input network.txt --verbose --log-file debug.log
```

---

## Migration from C++ LaNet-vi 3.x

LaNet-vi 5.0 is a complete Python rewrite that includes all features from the C++ version:

**Key Differences:**
- **Language**: C++ → Python 3.9+
- **Rendering**: POV-Ray → Matplotlib
- **Configuration**: Custom format → YAML
- **CLI**: Single-hyphen → Double-hyphen flags (Unix/GNU standard)
- **Dependencies**: No external renderers, pure Python stack

**Feature Parity:**
- ✅ K-core decomposition
- ✅ K-dense (m-core) decomposition
- ✅ D-core decomposition (directed graphs)
- ✅ Spiral/semicircular layouts
- ✅ Community detection
- ✅ Random graph generation
- ➕ Enhanced JSON exports
- ➕ Information theory metrics
- ➕ Spatial indexing
- ➕ Type-safe configuration

---

[5.0.0]: https://github.com/conexdat/lanet-vi/releases/tag/v5.0.0
