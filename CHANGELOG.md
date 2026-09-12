# Changelog

All notable changes to LaNet-vi will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security

- Refreshed `uv.lock`: resolves all 79 open Dependabot alerts (1 critical in
  jupyter-server, 42 high across Pillow, tornado, urllib3, mistune, jupyterlab and others).
  Only `requests` and Pillow (via matplotlib) are runtime dependencies; the rest were
  pulled in by the `dev` extra.
- New `Security` workflow: `pip-audit` over the locked tree on every lockfile change and
  weekly, plus `zizmor` over the workflow files.
- GitHub Actions pinned to commit SHAs, `persist-credentials: false` on every checkout,
  and no cache restore in the release workflow.
- `SECURITY.md` now states supported versions, the private reporting channel, response
  targets and the coordinated-disclosure process.

### Removed

- **Python 3.9 support.** 3.9 reached end of life in October 2025 and the patched
  releases of Pillow, jupyter-server and others require 3.10+. `requires-python` is now
  `>=3.10`; the code base uses `X | None` unions and `zip(..., strict=True)`.

### Fixed

- `write_decomposition_json` crashed when components were present (it read non-existent
  `Component.id` / `Component.index` attributes).
- `lanet_vi.community.base` failed to import on Python 3.9 because of `X | None` return
  annotations evaluated at runtime.
- `get_community_colors` returned RGBA quadruples and used the removed
  `matplotlib.cm.get_cmap` API; it now returns RGB triples via `pyplot.get_cmap`.
- `--community-algorithm` now rejects unknown values with a usage error instead of a
  Pydantic validation traceback.
- D-core decomposition (`--directed --decomp dcores`) crashed because `DecompositionResult`
  had no `metadata` field; the field now exists and holds the `(k_in, k_out)` pairs.
  `min_index` is now derived from the data instead of hard-coded to 1.
- `lanet_vi.community.base` used `X | None` annotations that fail to import on Python 3.9.
- `lanet_vi.__version__` reported `4.0.0`; it now reads the installed package version
  (`5.0.0`) from package metadata, so `pyproject.toml` is the single source of truth.

### Changed

- CAIDA examples, README and docs now use the 20251001 AS-relationships snapshot
  (78,370 ASes, 489,407 relationships, k-cores 1-149); example images regenerated.

### Infrastructure

- `main` is protected by a GitHub ruleset: changes land through pull requests with green CI
  and an automatically requested Copilot code review; force-pushes and deletions are blocked.
- mypy is now a blocking CI check (`disallow_untyped_defs`); the whole package type-checks
  cleanly. Pydantic models use `ConfigDict` instead of the deprecated inner `Config` class.
- CI split into `lint` (ruff check, ruff format, mypy), a `test` matrix on Python 3.9–3.13
  (Ubuntu) plus macOS 3.12, and `build` (uv build + twine check). Coverage is enforced
  with a minimum threshold and uploaded as an artifact.
- Ruff formatter adopted; `UP` (pyupgrade) and `B` (bugbear) rule sets enabled and the
  codebase migrated to PEP 585 built-in generics.
- pre-commit hooks (ruff lint/format, file hygiene) via `.pre-commit-config.yaml`.
- Dependabot for GitHub Actions and Python (uv) dependencies.
- PyPI releases use trusted publishing (OIDC) through the `pypi` environment instead of a
  long-lived API token.
- Added `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CODEOWNERS`, issue and
  pull request templates, and `AGENTS.md` (agent/contributor instructions; `CLAUDE.md`
  now includes it).
- New tests for the CLI, rendering pipeline, d-cores, writers and version metadata.
- Python 3.13 added to the supported versions.

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

[Unreleased]: https://github.com/CoNexDat/LaNet-vi/compare/v5.0.0...HEAD
[5.0.0]: https://github.com/CoNexDat/LaNet-vi/releases/tag/v5.0.0
