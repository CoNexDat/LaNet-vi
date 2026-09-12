# Migrating from the C++ LaNet-vi

LaNet-vi 5.x is a from-scratch Python rewrite of the C++ LaNet-vi (versions 1.x to 3.0.1,
2005–2016, by Mariano G. Beiró and J. Ignacio Alvarez-Hamelin, distributed on
[SourceForge](https://sourceforge.net/projects/lanet-vi/) under the Academic Free License 3.0).
This page maps the old command line onto the new one and states, feature by feature, what
was ported, what behaves differently and what is missing.

It is the outcome of a source-level comparison done in September 2026 between the C++ tree
(the released 3.0.1 sources, the later unreleased development tree that added d-cores, and
the 3.0.2/4.0.0 driver code) and `lanet_vi` 5.0.x. The 5.0.0 changelog entry says "all legacy
features included"; that was optimistic. Gaps and divergences found here are tracked as
GitHub issues, and this page is updated as they are closed.

## Which C++ code is the reference

| Tree                    | What it is                                                                                          |
|-------------------------|-----------------------------------------------------------------------------------------------------|
| LaNet-vi 3.0.1 (2016)   | Last public release on SourceForge. k-cores, k-denses, POV-Ray and SVG output, k-connectivity, Gomory-Hu connectivity. |
| 3.0.2 / 4.0.0 drivers   | Unreleased driver updates: `-directed` (d-cores, table output only, no image), `-fromlayer` for k-denses, PDF via SVG, a "json" engine that actually emits SVG. |
| Development library     | Unreleased library additions: `graph_dcores`, `espiral` (never called), community detection (`solution_lanci*`, `solution_submodular`), `erdos_renyi`, `mutual_information`. Only `graph_dcores` was linked into the 4.0.0 binary. |

The original C++ sources are not part of this repository. Get them from SourceForge.

## Command-line mapping

Status legend: **same** = same semantics; **differs** = accepted, but behaviour or default
differs (see notes); **inert** = accepted by the CLI or YAML config but has no effect today;
**missing** = no Python equivalent.

| C++ option (3.0.1 / 4.0.0)          | Python (`lanet-vi visualize`)              | Status  | Notes |
|-------------------------------------|--------------------------------------------|---------|-------|
| `-input <file>`                     | `--input`, `-i`                            | same    | Reader is stricter, see I/O below. |
| `-output <file>`                    | `--output`, `-o`                           | differs | C++ derived `<input>_<color>_<bg>_<WxH>[_kconn][_names]POV.png`; Python defaults to `output.png`. |
| `-decomp kcores\|kdenses`           | `--decomp kcores\|kdenses\|dcores`         | same    | `dcores` is new; the C++ `-directed` printed a table and no image. |
| `-directed`                         | `--directed`                               | differs | Python renders an image from `(k_in, k_out)` pairs. The CLI path currently crashes, see d-cores. |
| `-names <file>` / `-names` (no file) | `--names <file>`                          | inert   | Names are loaded but never drawn: the label switch is not exposed on the CLI. C++ `-names` without a file labelled nodes with their numbers; names could contain spaces. |
| `-font <value>`                     | `--font-zoom`                              | inert   | Only affects the label path above. |
| `-coresfile <file>`                 | `--cores-file <file>`                      | differs | The C++ option was dead; the real artefact was `log/cores.log` with `<index> <node>` lines. Python writes a CSV with a header (`node_id,<type>_index`) or JSON. |
| `-colorsFile <file>`                | `--colors-file <file>`                     | differs | Same `node r g b` format. C++ hid the colour legend and painted nodes absent from the file black/white; Python keeps the legend and shell colours. |
| `-coordDistributionAlgorithm classic\|pow\|log` | `--coord-distribution`         | inert   | The C++ selected between two layout procedures. The Python layout does not use it. |
| `-logfile`, `-logstdout`            | `--log-file`, `--verbose`, `--quiet`       | differs | Different facility; the C++ flags only captured renderer stderr (`-logstdout` was a no-op). |
| `-multigraph`                       | `--multigraph`                             | differs | Currently crashes for k-cores (`nx.core_number` rejects multigraphs). |
| `-weighted`                         | `--weighted`, `-w`                         | differs | See weighted k-cores below. |
| `-strengthsIntervals <method>`      | `--strength-intervals`                     | differs | Three computed methods match. `custom` with `-strengthsIntervalsFile` is missing. |
| `-maximumStrength <value>`          | YAML `decomposition.maximum_strength`      | differs | Not on the CLI. |
| `-granularity <n>`                  | `--granularity`                            | differs | C++ default is the maximum degree; Python caps the default at 100. |
| `-bckgnd white\|black`              | `--background`                             | same    | Default black in both (the C++ help text wrongly says white). |
| `-color col\|bw\|bwi`               | `--color-scheme`                           | differs | Colour stops differ; greyscale is inverted relative to C++. See rendering. |
| `-eps <value>`                      | `--epsilon`                                | differs | Default 0.18 in C++, 0.40 in Python. Only the ring-width envelope survives, see layout. |
| `-delta <value>`, `-gamma <value>`  | `--delta`, `--gamma`                       | inert   | Read by the config, not used by the layout. |
| `-fromlayer <k>`                    | `--from-layer`                             | inert   | C++ induced the subgraph of index ≥ k and recomputed with the parent p-function. |
| `-edges <0..1>`                     | `--edges-percent`                          | differs | Default 0.0 → 0.5; sampling algorithm differs, see rendering. |
| `-minedges <n>`                     | `--min-edges`                              | differs | Default 1000 → 50000. |
| `-W <px>`, `-H <px>`                | `--width`/`-W`, `--height`/`-H`            | differs | Default 800×600 → 2400×2400. Python crops to content (`bbox_inches="tight"`), so the PNG is not exactly W×H, and rejects aspect ratios outside 0.5–3. |
| `-window hs he vs ve`               | —                                          | missing | Viewport cropping. |
| `-u <value>`                        | YAML `visualization.unit_length`           | inert   | |
| `-net`                              | —                                          | missing | Never implemented in C++ either. |
| `-java`                             | —                                          | missing | Java viewer. |
| `-render povray\|svg\|json`         | YAML `renderer`                            | inert   | Only matplotlib exists; `networkx`/`plotly` values are accepted and ignored. Output format follows the file extension. |
| `-opacity <0..1>`                   | YAML `visualization.opacity`, `--edge-alpha` | differs | `opacity` is used only when `gradient_edges` is off; `--edge-alpha` is the live knob. |
| `-nocliques`                        | `--no-cliques`                             | differs | The central-core clique layout is not implemented, so the flag changes little. |
| `-drawCircles`                      | `--draw-circles`                           | differs | Python draws circles at synthetic positions on the shell ring, not around the laid-out components. |
| `-alpha <value>`, `-beta <value>`   | `--alpha`, `--beta`                        | inert   | Used only by the unreachable component-packing code. C++ default `alpha` 0.3, Python 1.0. |
| `-seed <n>`                         | `--seed`                                   | differs | Python seeds node placement (an improvement). Edge sampling still uses an unseeded RNG, so images are not reproducible. |
| `-ratioConstant <value>`            | YAML `layout.ratio_constant`               | inert   | |
| `-kconn`, `-kconntype strict\|wide` | —                                          | missing | k-connectivity analysis and its visual encoding. |
| `-connectivity`, `-innerConnectivity` | —                                        | missing | Gomory-Hu edge-connectivity analysis. |
| `-onlygraphic`, `-nographic`        | —                                          | missing | Tied to the external renderers. |
| `-colorScaleMaxValue <n>`           | `--color-scale-max`                        | differs | Honoured for k-cores; the C++ `+2` offset for m-cores is not applied. |
| `-showDegreeScale 0\|1`             | `--show-degree-scale`, `--show-size-legend` | differs | Mis-mapped: `show_degree_scale` toggles the colour legend; the degree/size legend is `show_size_legend`. Default-true flags cannot be turned off from the CLI. |
| `-measure kdense\|mcore`            | YAML `decomposition.measure`               | inert   | C++ default `mcore` labelled shells as k−2; Python always shows k. |
| `lanet.cfg` in the working directory | `--config file.yaml`                      | differs | C++ precedence was defaults < config file < command line. In Python `--config` replaces every other flag except `--weighted`/`--multigraph`. The YAML written by `lanet-vi config` cannot currently be loaded back. |

Python-only options: `--edge-alpha`, `--min-edge-width`, `--max-edge-width`,
`--node-size-scale`, `--node-edge-color`, `--gradient-edges`, `--legend-fontsize`,
`--show-size-legend`, the community flags, the spiral-layout flags, `--verbose`, `--quiet`,
`--log-file`, and the `config`, `info` and `generate` subcommands. Of these, the community
flags and the spiral-layout flags are currently inert.

## Feature parity by subsystem

### Decomposition

| Feature                         | Status  | Notes |
|---------------------------------|---------|-------|
| Unweighted k-cores              | same    | Same indices as the C++ peeling (Batagelj–Zaversnik via NetworkX). Directed input is silently treated as undirected where C++ refused it; self-loops and multigraphs crash where C++ handled them. |
| Weighted k-cores (p-function)   | differs | The three interval methods and `maximumStrength` match. The C++ then peeled, re-evaluating each vertex's strength over neighbours that are still in a higher shell; Python only bins total strength, so the result is a strength histogram, not a decomposition. Index base is also shifted by one for `equalNodesPerInterval` and `equalIntervalSize`. `--weighted` on a two-column file yields NaN weights. |
| k-dense / m-core                | differs | Same index formula (edge shell / 2 + 2). Two defects: triangle enumeration depends on adjacency insertion order and can miss triangles, and the dual graph is peeled with a plain vertex k-core instead of the C++ triangle-pair peeling (equivalent to k-truss). Results differ on about a fifth of random test graphs. |
| d-cores                         | differs | Python computes independent in-cores and out-cores per node, which is the older C++ `graph_dcores_old` algorithm; the later C++ computed a per-`l` table. The `Network`/CLI path currently crashes when building components. |
| `-fromlayer`                    | inert   | |
| Components                      | differs | Python components are the C++ "clusters" (connected components of one shell). The C++ nested component hierarchy (components of each k-core, recursively) is not modelled. |

### Layout

The C++ layout is the algorithm of Alvarez-Hamelin et al. (NIPS 2005) and Beiró et al.
(NJP 2008): nested components with their own centre and radius, ring radius from `eps`,
`delta`, `gamma` and `u`, radial position of a node from the shells of its neighbours,
angular position from the circular average of its higher-shell neighbours, the central core
decomposed into cliques placed on circular sectors, and components within a shell packed by
a circle-packing routine (`classic` versus `pow`/`log`).

| Mechanism                              | Status  | Notes |
|----------------------------------------|---------|-------|
| Concentric rings by shell              | differs | Python draws one ring per shell about the origin with a fixed radius of 80 units per shell step. `delta`, `gamma`, `u`, `ratioConstant` do not enter. |
| Radial position (NIPS 2005 formula 1)  | differs | The neighbour-based average was replaced by a uniform random draw; only the `(1−eps) + eps·average` envelope remains. |
| Angular position (circular average)    | differs | Ported, with a different averaging rule (vector mean instead of arc interpolation), an added order-dependent jitter, and angles measured from the origin rather than from the component centre. |
| Nested components, formulas (3)–(5)    | missing | |
| Component circle packing               | inert   | A port of `distribute_components` exists but is unreachable; it also lacks the growth loop and enlarges radii where the C++ shrank them. |
| Central-core clique decomposition      | missing | A faithful port of `placeInCircularSector` exists but has no caller; the top shell is placed on an evenly spaced small circle. |
| `-nocliques` sector placement          | differs | |
| k-dense layout variant                 | missing | k-denses and d-cores reuse the k-core layout. |
| Node size versus degree                | differs | Different formulas (with a switch at 1000 nodes), and the size legend uses yet another formula. |
| Spiral layout                          | inert   | Faithful port of `espiral.cpp`, which was never called in C++ either; `--use-spiral-layout` does nothing. |

### Rendering

| Feature                          | Status  | Notes |
|----------------------------------|---------|-------|
| Colour scale (`col`)             | differs | C++ ran magenta → blue → cyan → green → yellow → red (max core red, pastel offsets). Python runs blue → … → red → magenta (max core magenta). Classic LaNet-vi figures are not reproduced. |
| Greyscale (`bw`, `bwi`)          | differs | Inverted: C++ max core black, Python max core white. |
| Luminosity alternation           | differs | 0.7/1.2 in C++, 0.7/1.0 in Python. |
| k-dense colouring rules          | missing | `+2` on the scale maximum for m-cores, labels as k−2, alternation only on black. |
| Edge gradient (two halves, swapped colours) | same | |
| Edge sampling                    | differs | C++: per-edge Bernoulli with p = max(edges, minedges/E), reproducible. Python: shell-stratified quota topped up with core edges, unseeded. |
| Edge colour and width            | differs | Fixed 0.75 darkening for every scheme; linear width instead of degree-based radius; weights ignored. |
| Node borders                     | differs | C++ border = node colour; Python none unless `--node-edge-color`. |
| Colour legend                    | differs | Title hardcoded to "k-core" (issue #10); black outlines invisible on the default background; drawn even with a colours file. |
| Degree/size legend               | differs | Sizes do not match the drawn nodes; no strength variant. |
| Node labels                      | inert   | See `-names` above; no overlap avoidance. |
| Component border circles         | differs | Synthetic positions. |
| Exact `-W`/`-H` framing, `-window` | missing | |
| SVG, PDF via rsvg, POV-Ray, Java viewer | missing | matplotlib writes whatever format the extension implies. |
| k-connectivity visuals           | missing | |

### I/O

| Feature                     | Status  | Notes |
|-----------------------------|---------|-------|
| Edge list reader            | differs | C++ accepted any whitespace and ignored an unused weight column. Python accepts a single space only (tabs and repeated spaces fail), and an unweighted read of a three-column file with non-integer weights fails. |
| Repeated edges              | differs | C++ kept the first occurrence (simple graph); Python keeps the last weight. |
| Self-loops                  | differs | Crash in Python. |
| Edge list writer            | differs | `lanet-vi generate` writes tab-separated files that `lanet-vi visualize` cannot read. |
| Names, colours files        | differs | See the option table. |
| Cores file                  | differs | See the option table. |
| Compressed input, CAIDA reader, `#` comments | Python-only | |

### Analysis features

| Feature                                   | Status  | Notes |
|-------------------------------------------|---------|-------|
| k-connectivity (`-kconn`, strict/wide)    | missing | |
| Gomory-Hu edge connectivity, inner connectivity | missing | |
| Community detection                       | replaced | The C++ development tree had an LFK-style local community growth (`solution_lanci*`) and a submodular modularity agglomeration, never linked into the binary. Python wraps NetworkX Louvain and greedy modularity; the CLI flags are currently inert. |
| Partition metrics                         | partly  | Python NMI (`arithmetic`) equals the C++ `mutual_information`; MI, VI, ARI, overlap, Jaccard are Python-only; `busch_information` is not ported. |
| Random graphs                             | replaced | The C++ had an unlinked 12-line G(n,p) generator; Python wraps NetworkX generators with seeds. |

## Known defects found during the comparison

These are Python bugs independent of parity, listed so users know what to expect until the
tracking issues are closed:

- `--multigraph`, self-loops, tab-separated input, and the output of `lanet-vi generate`
  all abort the run.
- `--directed --decomp dcores` crashes when building components.
- `lanet-vi config` writes YAML that `--config` cannot load, and `--config` discards the
  other command-line flags.
- `--names` never draws labels; default-true boolean flags cannot be switched off.
- k-dense indices depend on edge order in the input file.
- Images are not reproducible with `--seed` because edge sampling is unseeded.
- Legend sizes and titles do not describe the drawn nodes.
