# Changelog

All notable changes to LaNet-vi will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Documentation

- Documentation site at <https://conexdat.github.io/LaNet-vi/> (#12): MkDocs with the
  Material theme, the existing guides, the changelog and an API reference generated from
  the docstrings by mkdocstrings. `mkdocs build --strict` runs in CI on every PR;
  `.github/workflows/docs.yml` deploys to GitHub Pages on every push to `main`. The
  `docs` extra installs the tooling; `uv run mkdocs serve` previews the site.
- Content pass on the docs: `docs/usage.md` is now the reference for every CLI option
  (default, meaning, C++ equivalent, inert ones marked), the configuration file, the input
  and output formats and the Python API; new `docs/cpp-migration.md` translates every
  LaNet-vi 3.x flag; the README is shorter, drops the claims that did not hold (rendering
  "millions of nodes", community detection as a CLI feature) and links to the guides;
  `docs/concepts.md` cites the actual papers (NIPS 2005, NJP 2008, D-cores, k-truss).
  Every shell block in the docs is now a command that runs as written (the multi-line
  ones had `\\` continuations that broke when pasted).

### Added

- `--window HSTART HEND VSTART VEND` (`window` in YAML), the C++ `-window`: render only
  that fraction of the frame, measured from the top-left corner, at the full pixel size
  (#25). PDF and SVG output (by the `--output` extension) already worked and is now
  documented.

## [5.1.0] - 2026-09-18

The first release after a source-level comparison with the original C++ LaNet-vi 3.x. The
5.0.0 rewrite had replaced most of the algorithms with approximations; 5.1.0 ports them
faithfully, so the pictures are the LaNet-vi pictures again: the placement of
Alvarez-Hamelin, Dall'Asta, Barrat & Vespignani (NIPS 2005), the triangle peeling of
k-dense, the weighted k-core peeling, the color scale of `types.cpp` with red for the top
core, the per-edge sampling seeded by `--seed`, and the `pow`/`log` circle packing of
sibling components. It also fixes the crashes and CLI problems found on the way (d-cores through
the CLI, tab-separated input, self-loops, `--multigraph`, YAML round trip, `--config`
precedence, `--names`), drops Python 3.9, resolves every open security alert and adds
`CITATION.cff`. The details, by category:

### Documentation

- American English is now the mandatory spelling for identifiers, docstrings, comments
  and docs (`AGENTS.md`, `CONTRIBUTING.md`, PR template). Existing British spellings in
  comments, docstrings and docs were corrected; the only renamed identifiers are private
  (`_Placer.place_centre` → `place_center`) and six test names, so no public API changed.
- `CITATION.cff` (validated with `cffconvert`): software citation with the package
  authors, version and release date, plus the NIPS 2005 and New Journal of Physics 2008
  papers as references; the release checklist now updates it with each version.
- README now has a Heritage section crediting the original C++ LaNet-vi (Beiró,
  Alvarez-Hamelin et al., 2005–2016), its SourceForge distribution under the Academic Free
  License 3.0 and the original homepage; the License section states the relation between
  the MIT-licensed rewrite and the AFL-licensed original.
- The 5.0.0 entry below overstated feature parity with the C++ version; it now carries a
  note pointing to the tracking issues (#18–#26).

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

### Added

- The `pow` and `log` coordinate distributions of the C++ (`--coord-distribution`, #18):
  the network is a unit disc, the children of a component share a disc of radius
  `((T - S) / T)^(1/8) R` capped at `0.96 R` (`log`: `sqrt((n - s) / n) R`) and sibling
  components are packed as non-overlapping discs whose areas follow
  `alpha w^beta` of their `sum(log(1 + d)^(2 / beta))` share, inflated 1 % per round
  until they no longer fit (`distribute_components.cpp`, seeded by `--seed`). Node radii
  in these modes are `0.007 ratio_constant log(1 + d)^1.5` (`computeHostRatio`), with the
  new `--ratio-constant` (C++ `-ratioConstant`, default 1). K-dense pictures use the
  variant of `kdenses_component.cpp` (fixed `0.92` shrink, same-index neighbors in
  formula (1) and the angle, `epsilon` scaled by `tau`, no `u`, node radius
  `ratio_constant sqrt(log(1 + d))` with `ratio_constant` auto-adjusted by the top cores).
  `--alpha` and `--beta` now act; `alpha` defaults to the C++ 0.3 (was an inert 1.0).
  Deliberate deviations: inflated discs keep their angle and are pulled towards their
  container's center (the C++ pulled them towards the picture's origin, and overwrote
  ``x`` before computing ``y`` from it, skewing the angle), unplaced neighbors are skipped
  in the angle. `classic` stays the default for every decomposition, although the C++ k-dense
  only ever had this placement (its default flags behaved as `pow`).

### Deprecated

- `use_spatial_hashing` is accepted and ignored: the circle packing follows the C++
  algorithm (#18).
- `edge_alpha` (`--edge-alpha`) is an alias of `opacity` (`--opacity`, the C++
  `-opacity`, default 0.2); `min_edge_width` / `max_edge_width` (`--min-edge-width`,
  `--max-edge-width`) are accepted and ignored (#24). `lanet-vi config` no longer writes
  the deprecated aliases (`edge_alpha`, `show_size_legend`) into the template, where they
  were ignored next to the current field.

### Removed

- `visualization.circular_average` and `layout.compute_hierarchical_layout` (replaced by
  `lanet_layout`); `layout.distribute_components` is now the faithful port used by the
  `pow`/`log` modes (the earlier version lacked the growth loop) and `layout.SpatialHashGrid`
  is gone with it.
- **Python 3.9 support.** 3.9 reached end of life in October 2025 and the patched
  releases of Pillow, jupyter-server and others require 3.10+. `requires-python` is now
  `>=3.10`; the code base uses `X | None` unions and `zip(..., strict=True)`.

### Fixed

- Rendering follows the C++ LaNet-vi again (#24). Color scale from `types.cpp`: the
  rainbow runs magenta → blue → cyan → green → yellow → **red** (the maximum index was
  magenta), the black-and-white scale runs white → gray → **black** (it was inverted) and
  `bwi` interlaces that same scale; consecutive shells alternate a luminosity of 0.7 and
  1.2 (was 0.7/1.0), with the k-dense rules of `graphics_kdenses.cpp` (constant 0.9 on a
  white background; with `measure = mcore`, the default, `--color-scale-max` is read in
  m-core units and the legend labels each k-dense as `k - 2` under the title `m-core`;
  `--measure kdense` keeps the k-dense numbers). Nodes absent from a `--colors-file` are
  white on black / black on white, and the color legend is hidden in that case.
- Edges: every edge is kept with probability `max(edges_percent, min_edges / E)` (the C++
  per-edge Bernoulli) using the layout `seed`, so `--seed` makes pictures reproducible
  (the stratified sampler used the unseeded `random` module). Edge colors darken by 0.75
  in `col` pictures and lighten by 1.2 in `bw`/`bwi` ones; k-dense edges are one color,
  their own dense index darkened by 0.5 (the flat gray the 3.0.1 release used for edges
  between clusters was removed in the 3.0.2 and 4.0.0 drivers and is not reproduced). Edge width is 0.2 host radii of the smaller
  endpoint degree (the C++ `ratioEdge`), in layout units with a one-pixel floor; edges are
  drawn under the nodes in increasing index order. Nodes are opaque (the >1000-node path
  drew them at alpha 0.9).
- Legends are drawn in layout units where `generateNetworkFile` put them, so they scale
  with the picture: the color legend has one circle per index (from 1, or 2 for
  k-denses), labeled in the index color every `max // 15 + 1` indices; the degree legend
  shows `ceil(dmax / 4^i)` (down to 2, at most five) with the radius those nodes have in the
  picture (it used its own size formulas and ignored `node_size_scale`), or strengths for
  weighted graphs. Weighted graphs whose strengths never exceed 1 (where the C++ strength
  law divides by `log(max) <= 0`) now size nodes by the degree law instead of a constant
  radius, and the legend follows.
- The PNG is exactly `width x height` pixels (`bbox_inches="tight"` cropped it); the frame
  is scaled uniformly to fit and centered, so any aspect ratio works and the validator that
  rejected sizes such as 3200x800 is gone.
- Weighted k-cores (#20) now peel, as the C++ `findCores` weighted branch does: every
  node starts in the strength interval of its total strength and, when a shell is
  removed, its neighbors are re-binned using only the strength they still receive from
  nodes above that shell. The previous code only binned total strengths (a histogram,
  not a core decomposition; it disagreed with the C++ on 1,570 of the 2,070 random
  configurations checked, see PR #31). Default granularity is the maximum degree again (the cap at 100 is
  gone). Indices run `1..granularity` in every interval method (0 for isolated nodes);
  the C++ 3.0.1 ran `2..granularity+1` in `equalIntervalSize` and
  `equalNodesPerInterval` because of a duplicated `0.0` boundary. `equalLogIntervalSize`
  starts from the smallest positive strength instead of dividing by zero. The remaining
  strength of each node is kept incrementally while peeling (the C++ re-summed it on
  every re-binning), and negative weights are refused with a clear error.
- New `--maximum-strength` (config `maximum_strength`, the C++ `-maximumStrength`) and
  `--strength-intervals custom` with `--strength-intervals-file` (config
  `strength_intervals_file`, the C++ 4.0.0 `-strengthsIntervalsFile`) (#20).
- Invalid configuration (from the YAML file or the flags), an unusable
  `--strength-intervals-file` and inputs the decomposition refuses (negative weights,
  `--decomp dcores` on an undirected graph) are reported as usage errors instead of
  tracebacks.
- K-dense decomposition (#19) now implements the C++ triangle-pair peeling (the k-truss
  decomposition): every edge starts with its triangle count and removing an edge lowers the
  count of the two other sides of each triangle it closed. The previous code took a plain
  vertex k-core of the edge/triangle dual graph, which over-estimates the index of edges
  whose triangles share a side, and its triangle enumeration assumed sorted adjacency, so
  it silently missed triangles on real edge lists. Parallel edges and self-loops are
  ignored. Per-edge indices are exposed as `result.metadata["edge_indices"]` (JSON export
  writes them as `[u, v, index]` triples). Verified against a brute-force k-truss on random
  graphs. The CAIDA 20251001 example now spans k-dense 2–105 (was 2–55); its image is
  regenerated.
- The color legend is titled after the decomposition (`k-dense`, `d-core`) instead of
  always `k-core` (#10).
- `--directed --decomp dcores` crashed with a Pydantic `ValidationError` while building
  components (`Component` was constructed with `id`/`index` instead of `component_id`/
  `shell_index`). The d-core path through `Network.decompose()` and the CLI works again
  and is now covered by tests (#21).
- Edge-list reader (#22): columns may be separated by any whitespace (tabs and repeated
  spaces used to abort with `invalid literal for int()`); an unused third column is
  ignored when the graph is not weighted; `--weighted` on a two-column file uses weight
  1.0 instead of NaN; self-loops are dropped with a warning instead of crashing the
  decomposition; non-integer node ids and single-column files give a clear error. All of
  this matches the C++ reader.
- `--multigraph` no longer crashes k-cores: parallel edges count towards the degree (the
  C++ behavior) and, for weighted multigraphs, their weights are summed into the strength.
- `write_edge_list` (and therefore `lanet-vi generate`) now defaults to space-separated
  output; its files were tab-separated and could not be read back by `lanet-vi visualize`.
- Node names may contain spaces (the name is the rest of the line, quotes stripped) and
  the colors file accepts tabs.
- `lanet-vi config` wrote YAML with `!!python/object/apply` tags that `--config` could
  not load; enums are now written as plain strings (#23).
- `--config` no longer discards the other command-line flags. Precedence is now the C++
  one: defaults < YAML file < flags given explicitly on the command line (#23).
- `--names` now draws the labels (it loaded them and drew nothing); use
  `--no-node-labels` to load names without drawing them. A node named `0` is no longer
  skipped (#23).
- `--show-degree-scale` toggled the color legend instead of the degree legend. It now
  controls the degree (node size) legend, as the C++ `-showDegreeScale`; the color legend
  has its own `--show-color-legend/--no-show-color-legend` (config `show_color_legend`).
  `show_size_legend` is kept as a deprecated alias that folds into `show_degree_scale` (#23).
- Default-true boolean flags (`--show-degree-scale`, `--gradient-edges`,
  `--color-by-community`, `--draw-community-boundaries`) now have `--no-...` forms (#23).
- Options that the current implementation does not use (`--from-layer`,
  `--use-spiral-layout` and the `--spiral-*` settings, `--detect-communities` and the
  other community flags) say so in their help text, with the tracking issue (#18, #23).
  (`--delta`, `--gamma`, `--alpha`, `--beta` and `--coord-distribution`, inert at the
  time, act since the layout ports below.)
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

- The degree legend stacks its rows so that a sample never overlaps the previous one
  (needed in the `pow` / `log` modes, whose radii reach a fifth of the unit disc) and
  keeps the C++ text size instead of growing with the row pitch; unchanged on large
  classic pictures, where the C++ pitch is the larger.
- **Layout (#18): the actual LaNet-vi placement is back.** `visualization/lanet_layout.py`
  ports `kcores_component.cpp` / `graph_kcores_components.cpp` (classic mode): nested
  components (connected pieces of the inner core, recursively) with their own center,
  radius and scale (formulas (3)-(5) of NIPS 2005), rings one unit apart with the top core
  as a disc whose radius follows its log-degrees, node radius from the depth of its
  higher-index neighbors (formula (1)), angle from the circular average of their angles,
  top cores split into cliques laid along U-shaped paths in angular sectors, and the
  `--no-cliques` sector formula (2). The previous code placed every shell on a fixed
  80-unit ring at a random radius and measured neighbor angles from the origin. K-dense
  and d-core results use the same classic placement, building the component tree with
  their own edge index (the edge dense index, as `kdenses_component.cpp` walks it); the
  rest of that file's variant (`>=` neighbors, `tau`, sibling circle packing,
  `ratioConstant` radii) belongs to the C++ `pow`/`log` mode, ported afterwards (see
  "Added" above). `epsilon`, `delta`, `gamma`,
  `unit_length`, `seed`, `--no-cliques` and `--draw-circles` now do what the C++ flags
  did; `epsilon` defaults to the C++ 0.18 again. The picture is framed as the C++
  viewport (1.6 x 1.2 times the network radius), leaving the margin the legends sit in.
- Node radii follow the C++ `computeHostRatio` (`0.4 (log(1+d)/log(dmax))^0.7` layout
  units, strength-based for weighted graphs) and are drawn in layout units for graphs of
  any size (an `EllipseCollection` replaces the point-sized scatter), floored at one
  pixel so peripheral nodes stay visible. `node_size_scale` is a plain multiplier and
  defaults to 1.0.
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

> **Note (September 2026):** the parity claims in this entry were later found to be
> overstated. Several features listed as ported are missing, inert or behave differently;
> see issues #18–#26 for the current state.

Complete Python refactor of LaNet-vi 3.x (C++) with all legacy features included. This version brings all functionality from the latest C++ codebase (previously in `legacy/Source/`) into a modern Python implementation using NetworkX, pandas, and matplotlib.

### Core Features (Ported from C++ LaNet-vi 3.x)

- **K-core decomposition**: Classic k-core algorithm using NetworkX
- **K-dense decomposition**: Triangle-based decomposition (m-cores)
- **D-core decomposition**: Directed graphs with (k_in, k_out) pairs per node
  - Ported from `legacy/Source/graph_dcores.cpp`
  - CLI: `--directed --decomp dcores`
- **Spiral/semicircular layout**: Mathematical spiral placement using Newton-Raphson solver
  - Inspired by `legacy/Source/espiral.cpp`, which was never linked into a C++ release;
    the option is accepted but inert (#18)
  - CLI: `--use-spiral-layout`
- **Community detection**: Louvain and greedy modularity algorithms
  - A NetworkX-based replacement, not a port: the C++ community code
    (`solution_lanci*.cpp`, `solution_submodular.cpp`) was research code never linked
    into a LaNet-vi binary, and its local-growth and submodular methods are not in Python
    (#26)
  - CLI: `--detect-communities`
- **Community visualization**: Color-coded nodes with boundary overlays
- **Random graph generation**: Testing and benchmarking utilities
  - Erdős-Rényi, Barabási-Albert, Watts-Strogatz, Powerlaw cluster
  - NetworkX wrappers; the C++ `erdos_renyi.cpp` was a 12-line unlinked G(n, p) loop
    (#26)
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

LaNet-vi 5.x is a complete Python rewrite of the C++ version. Parity was overstated in
5.0.0 and restored piece by piece in 5.1.0 (see the 5.1.0 entry and issues #18–#26):

**Key Differences:**
- **Language**: C++ → Python 3.10+
- **Rendering**: POV-Ray / SVG → Matplotlib
- **Configuration**: Custom format → YAML
- **CLI**: Single-hyphen → Double-hyphen flags (Unix/GNU standard)
- **Dependencies**: No external renderers, pure Python stack

**Feature Parity (as of 5.1.0):**
- ✅ K-core decomposition, including the weighted peeling
- ✅ K-dense (m-core) decomposition (triangle-pair peeling)
- ✅ D-core decomposition (directed graphs)
- ✅ The LaNet-vi placement: `classic`, `pow` and `log` coordinate distributions
- ✅ Color scale, grayscale, legends, edge sampling
- ⚠️ Spiral/semicircular layout: option accepted but inert; never in a C++ release (#18)
- ⚠️ Community detection: NetworkX Louvain / greedy modularity, a replacement, not a
  port; the CLI flags are not wired into rendering yet (#23, #26)
- ⚠️ Random graph generation: NetworkX wrappers (#26)
- ❌ Not ported: k-connectivity, Gomory-Hu connectivity, SVG/PDF/POV-Ray output,
  `-window` cropping (#25)
- ➕ Enhanced JSON exports
- ➕ Information theory metrics
- ➕ Spatial indexing
- ➕ Type-safe configuration

---

[Unreleased]: https://github.com/CoNexDat/LaNet-vi/compare/v5.1.0...HEAD
[5.1.0]: https://github.com/CoNexDat/LaNet-vi/compare/v5.0.0...v5.1.0
[5.0.0]: https://github.com/CoNexDat/LaNet-vi/releases/tag/v5.0.0
