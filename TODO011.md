Some pending things that appear in the CI

Run uv run ruff check src/ tests/
N803 Argument name `spiral_K` should be lowercase
   --> src/lanet_vi/cli.py:135:5
    |
133 |         False, "--use-spiral-layout", help="Use spiral layout algorithm"
134 |     ),
135 |     spiral_K: float = typer.Option(10.0, "--spiral-K", help="Spiral scaling constant"),
    |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
136 |     spiral_beta: float = typer.Option(1.5, "--spiral-beta", help="Spiral tightness parameter"),
137 |     spiral_separation: float = typer.Option(
    |

D301 Use `r"""` if any backslashes in a docstring
   --> src/lanet_vi/cli.py:393:5
    |
391 |       log_file: Optional[Path] = typer.Option(None, "--log-file", help="Log to file"),
392 |   ) -> None:
393 | /     """
394 | |     Generate random graphs for testing and demonstration.
395 | |
396 | |     Examples
397 | |     --------
398 | |         # Erdős-Rényi with edge probability
399 | |         lanet-vi generate --output er.txt --model erdos-renyi --nodes 1000 --probability 0.01
400 | |
401 | |         # Erdős-Rényi with fixed number of edges
402 | |         lanet-vi generate --output er.txt --model erdos-renyi --nodes 1000 --edges 5000
403 | |
404 | |         # Barabási-Albert scale-free network
405 | |         lanet-vi generate --output ba.txt --model barabasi-albert --nodes 1000 --edges 3
406 | |
407 | |         # Watts-Strogatz small-world network
408 | |         lanet-vi generate --output ws.txt --model watts-strogatz --nodes 1000 \\
409 | |             --neighbors 6 --rewire 0.3
410 | |
411 | |         # Powerlaw cluster graph
412 | |         lanet-vi generate --output pc.txt --model powerlaw-cluster --nodes 1000 \\
413 | |             --edges 3 --triangle-prob 0.5
414 | |     """
    | |_______^
415 |       from lanet_vi.generators import (
416 |           generate_barabasi_albert,
    |
help: Add `r` prefix

No fixes available (6 hidden fixes can be enabled with the `--unsafe-fixes` option).
Error: Process completed with exit code 1.