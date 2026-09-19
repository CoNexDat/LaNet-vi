# LaNet-vi

**Large-scale network visualization by k-core decomposition.**

LaNet-vi draws large networks so that their structure is readable at a glance: nodes are
placed in concentric rings by their k-core (or k-dense, or d-core) index, the densest core
at the center, with node size following the degree and colors following the index. It is
the Python version of the C++ LaNet-vi that produced the well-known Internet AS-level maps
(Alvarez-Hamelin, Dall'Asta, Barrat & Vespignani, NIPS 2005; Beiró, Alvarez-Hamelin &
Busch, New J. Phys. 2008), and since 5.1.0 it follows the original algorithms.

<p align="center">
  <img src="https://raw.githubusercontent.com/CoNexDat/LaNet-vi/main/examples/outputs/caida_as_relationships_kcores.png" width="48%" alt="CAIDA AS relationships, k-cores">
  <img src="https://raw.githubusercontent.com/CoNexDat/LaNet-vi/main/examples/outputs/caida_as_relationships_kdenses.png" width="48%" alt="CAIDA AS relationships, k-denses">
  <br>
  <em>The Internet at the AS level (CAIDA AS relationships, October 2025: 78,370 ASes,
  489,407 links). Left: k-cores 1–149, Tier-1 and hypergiant networks in the red core.
  Right: k-denses (m-cores), the triangle-based decomposition.</em>
</p>

## Installation

```bash
pip install lanet-vi
```

Python 3.10 or newer. The package installs the `lanet-vi` command.

## Quick start

An edge list is a text file with one edge per line, `source target` (and an optional
weight as third column):

```bash
lanet-vi visualize --input network.txt --output network.png
```

The same from Python:

```python
import networkx as nx
from lanet_vi import LaNetConfig, Network

G = nx.Graph(nx.karate_club_graph().edges())
net = Network(G, LaNetConfig())
net.decompose()
net.visualize("karate.png")
```

## Where to go next

- **[Usage guide](usage.md)** — every CLI option, the configuration file, the input and
  output formats, the Python API step by step.
- **[Visualization guide](visualization.md)** — how the picture is built: placement,
  colors, sizes, edges, legends.
- **[Concepts](concepts.md)** — k-cores, weighted k-cores, k-denses, d-cores, and the
  papers behind them.
- **[Coming from the C++ LaNet-vi](cpp-migration.md)** — the flag translation for users
  of LaNet-vi 3.x and what differs.
- **[API reference](api.md)** — generated from the docstrings.
- **[Examples](https://github.com/CoNexDat/LaNet-vi/tree/main/examples)** — the CAIDA
  scripts behind the pictures above.

## Citing

If LaNet-vi is useful in your research, please cite the papers that introduced the
method and the software (see [`CITATION.cff`](https://github.com/CoNexDat/LaNet-vi/blob/main/CITATION.cff)
for a machine-readable version):

- J. I. Alvarez-Hamelin, L. Dall'Asta, A. Barrat, A. Vespignani. *Large scale networks
  fingerprinting and visualization using the k-core decomposition.* NIPS 2005.
- M. G. Beiró, J. I. Alvarez-Hamelin, J. R. Busch. *A low complexity visualization tool
  that helps to perform complex systems analysis.* New J. Phys. 10 (2008) 125003.

## Heritage and license

LaNet-vi 5.x is a from-scratch Python rewrite, MIT-licensed. The original C++ LaNet-vi
was developed since 2005 by Mariano G. Beiró and J. Ignacio Alvarez-Hamelin (Universidad
de Buenos Aires / CONICET) together with Alain Barrat, Luca Dall'Asta and Alessandro
Vespignani; its releases are on [SourceForge](https://sourceforge.net/projects/lanet-vi/)
under the Academic Free License 3.0. The
[README](https://github.com/CoNexDat/LaNet-vi#readme) has the full credits and the
relation between the two licenses.
