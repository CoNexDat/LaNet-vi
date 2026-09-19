# Coming from the C++ LaNet-vi 3.x

LaNet-vi 5.x is a Python rewrite of the C++ LaNet-vi (3.0.1, the last SourceForge
release, plus the `-directed` and `-strengthsIntervalsFile` options of the unreleased
4.0.0). Since 5.1.0 the decompositions, the placement and the rendering follow the C++
sources, so a picture made with the same options looks the same up to the random choices
(the random generators differ, so the same `-seed` does not give the same random draws).

What is different by design:

- **Rendering.** POV-Ray and the C++ SVG writer are replaced by matplotlib: `--output`
  decides the format by its extension (PNG, PDF, SVG). There is no `-render`, `-java`,
  `-onlygraphic` or `-nographic`; the picture is always produced.
- **Flags** use the GNU form (`--edges-percent`, not `-edges`) and can also come from a
  YAML file (`lanet-vi config`, `--config`), with the C++ precedence: defaults < file <
  explicit flags.
- **Defaults** are the C++ ones except where noted below (`--edges-percent 0.5`,
  `--min-edges 50000`, `--background black`, 2400 × 2400 pixels; the C++ drew 0 % of the
  edges with a floor of 1000, on white, at 800 × 600).
- **Logs.** `log/cores.log`, `log/kconn.log`, `log/gomory_hu*.log` do not exist;
  `--cores-file` writes the decomposition (CSV or JSON) and `--log-file` the run log.

## Flag translation

| C++ 3.x | LaNet-vi 5.x | Notes |
|---|---|---|
| `-input FILE` | `--input FILE` | `.gz` / `.bz2` are read directly |
| `-output FILE` | `--output FILE` | PNG, PDF or SVG by extension |
| `-decomp kcores\|kdenses` | `--decomp kcores\|kdenses\|dcores` | |
| `-directed` (4.0) | `--directed` | needed for `dcores` |
| `-measure mcore\|kdense` | `--measure mcore\|kdense` | k-dense legend numbering |
| `-names FILE` | `--names FILE` | labels are drawn when the file is given (`--no-node-labels` to keep them off) |
| `-font Z` | `--font-zoom Z` | |
| `-coresfile FILE` | `--cores-file FILE` | the C++ option was dead; the decomposition went to `log/cores.log` |
| `-colorsFile FILE` | `--colors-file FILE` | same format `node r g b` |
| `-weighted` | `--weighted` | |
| `-multigraph` | `--multigraph` | |
| `-strengthsIntervals M` | `--strength-intervals M` | same four methods |
| `-strengthsIntervalsFile FILE` (4.0) | `--strength-intervals-file FILE` | |
| `-maximumStrength S` | `--maximum-strength S` | |
| `-granularity N` | `--granularity N` | default: maximum degree, as the C++ (no cap at 100) |
| `-fromlayer K` | `--from-layer K` | accepted, not implemented yet (#23) |
| `-bckgnd white\|black` | `--background white\|black` | default is `black` here |
| `-color col\|bw\|bwi` | `--color-scheme col\|bw\|bwi` | |
| `-colorScaleMaxValue K` | `--color-scale-max K` | |
| `-showDegreeScale 0\|1` | `--show-degree-scale` / `--no-show-degree-scale` | plus `--show-color-legend` for the index legend |
| `-eps E` | `--epsilon E` | 0.18 |
| `-delta D` | `--delta D` | 1.3 |
| `-gamma G` | `--gamma G` | 1.5 |
| `-u U` | (YAML `visualization.unit_length`) | no CLI flag |
| `-coordDistributionAlgorithm classic\|pow\|log` | `--coord-distribution classic\|pow\|log` | |
| `-alpha A`, `-beta B` | `--alpha A`, `--beta B` | 0.3, 1.0 |
| `-ratioConstant C` | `--ratio-constant C` | auto-adjusted when absent, as the C++ |
| `-seed N` | `--seed N` | also seeds the edge sample |
| `-nocliques` | `--no-cliques` | |
| `-drawCircles` | `--draw-circles` | |
| `-edges P` | `--edges-percent P` | default 0.5 (C++: 0.0) |
| `-minedges N` | `--min-edges N` | default 50000 (C++: 1000) |
| `-opacity O` | `--opacity O` | 0.2 |
| `-W PX`, `-H PX` | `--width PX`, `--height PX` | any aspect ratio |
| `-window HS HE VS VE` | `--window HS HE VS VE` | |
| `-render povray\|svg` | — | matplotlib; see `--output` |
| `-java`, `-onlygraphic`, `-nographic`, `-net` | — | not applicable |
| `-logfile`, `-logstdout` | `--log-file FILE`, `--verbose` | |
| `-kconn`, `-kconntype strict\|wide` | — | not ported (#25) |
| `-connectivity`, `-innerConnectivity` | — | not ported (#25) |

New in 5.x, without a C++ equivalent: `--config`, `--node-size-scale`,
`--node-edge-color`, `--gradient-edges`, `--legend-fontsize`, `--show-color-legend`,
`--node-labels`, `lanet-vi info`, `lanet-vi generate`, and the Python API.

## What is not there

Tracked in [issue #25](https://github.com/CoNexDat/LaNet-vi/issues/25): k-connectivity
(`-kconn`), Gomory-Hu connectivity (`-connectivity`), POV-Ray scenes. The community
detection and random graph generators of the Python package are NetworkX-based
replacements, not ports of the authors' research code (#26). The spiral layout of the
development tree was never part of a C++ release and is inert here (#18).
