# AGENTS.md — LaNet-vi

Instructions for AI coding agents (Claude Code, Copilot, Codex, Cursor, ...) working in
this repository. `CLAUDE.md` includes this file. Humans: see `CONTRIBUTING.md`.

## What this project is

LaNet-vi is a Python package (`lanet_vi`, published on PyPI as `lanet-vi`) that
visualizes large networks by k-core, k-dense (m-core) and d-core decomposition, laying
nodes out in concentric rings and rendering with matplotlib. Version 5.x is a complete
rewrite of the C++ 3.x tool; the original C++ sources are kept locally under `legacy/`
(gitignored) only as provenance and are **not** part of the package.

## Layout

```
src/lanet_vi/
  cli.py                 Typer CLI: `lanet-vi visualize|config|info|generate`
  core/network.py        Network: load → decompose() → compute_layout() → visualize()
  decomposition/         kcores.py, kdenses.py, dcores.py → DecompositionResult
  visualization/         layout.py (ring layout), matplotlib_renderer.py, colors.py,
                         spiral_layout.py, spatial_index.py, community_viz.py
  io/                    readers.py (edge lists, CAIDA), writers.py (CSV/JSON), config_loader.py (YAML)
  models/                config.py (Pydantic settings), graph.py (result/layout models)
  metrics/, community/, generators/   partition metrics, Louvain/greedy, random graphs
tests/                   pytest; conftest.py has shared fixtures
docs/                    concepts.md, usage.md, visualization.md (plain markdown)
examples/                CAIDA AS-relationship examples + tracked output PNGs
```

Data flow: `Network.from_edge_list()` → `Network.decompose()` (fills
`DecompositionResult`: `node_indices`, `components`, `max_index`) →
`Network.compute_layout()` (`VisualizationLayout`) → `render_network()`.

## Commands

```bash
uv sync --all-extras                       # environment (Python 3.10–3.13 supported)
uv run pre-commit install                  # once per clone
uv run ruff check src/ tests/ examples/    # lint
uv run ruff format src/ tests/ examples/   # format (CI checks with --check)
uv run mypy src/lanet_vi                   # types (strict: disallow_untyped_defs)
uv run pytest                              # tests + coverage gate (see pyproject addopts)
uv build                                   # sdist + wheel via uv_build
uv run lanet-vi visualize --input edges.txt --output out.png
```

## Conventions

- Python 3.10 compatible: `X | None` unions are fine, `match` is fine, nothing newer
  (no `Self`, no `except*`). Ruff target is `py310`; it will flag anything newer.
- Line length 100. NumPy-style docstrings on public API (ruff `D`, convention `numpy`).
- All settings live in Pydantic models in `models/config.py`; the CLI builds a
  `LaNetConfig` from flags. New options go there first, then the CLI, then docs.
- Type hints on every function. mypy is part of CI.
- Tests: one file per module (`tests/test_<module>.py`), plain functions, fixtures in
  `tests/conftest.py`. Use `nx.karate_club_graph()` for small realistic graphs and
  `tmp_path` for any file output. Force `matplotlib.use("Agg")` in rendering tests.
- Keep `CHANGELOG.md` current: add a bullet under `[Unreleased]` for every user-visible
  change. Version lives only in `pyproject.toml`.
- Do not commit generated outputs (`*.png`, `*.csv`) outside `examples/outputs/`.
- Do not edit `uv.lock` by hand; run `uv lock` / `uv add`.

## Pull request workflow (required)

`main` is protected by a ruleset: no direct pushes, PR required, CI checks required,
Copilot code review requested automatically, review threads must be resolved.

**Repository policy: every PR must get a Copilot code review, and no PR is merged until
CI is green and Copilot has given the green light** (no unresolved Copilot findings). The
ruleset requests the review automatically when Copilot is enabled on the org; until then,
or if the automatic request does not appear within a few minutes, request it manually:

```bash
gh pr edit <n> --add-reviewer Copilot     # or the "Reviewers" gear in the PR sidebar
```

Re-request after every push (`gh pr edit <n> --add-reviewer Copilot` again, or the
re-request icon next to Copilot in the sidebar). Note: the ruleset only auto-requests on
PRs whose base is `main`; stacked PRs (base = another branch) always need a manual request.
If the request is refused or silently dropped (the sidebar shows Copilot greyed out with
"Monthly limit reached", or the API returns "could not resolve user"), stop and tell the
maintainer; do not merge without the review. The free Copilot tier has a small monthly
quota of code reviews, so keep PRs few and batch pushes where you can.

Concretely:

1. Branch from `main` (`feat/…`, `fix/…`, `docs/…`, `chore/…`), commit, push, open the
   PR with `gh pr create`. Fill the PR template checklist honestly.
2. Wait for CI: `gh pr checks <n> --watch`. If anything is red, read the log
   (`gh run view <run-id> --log-failed`), fix locally, push, and wait again.
3. Make sure a Copilot review is requested (automatic, or manually as above) and wait for
   it to appear. Read every comment:
   ```bash
   gh pr view <n> --comments
   gh api repos/CoNexDat/LaNet-vi/pulls/<n>/reviews
   gh api repos/CoNexDat/LaNet-vi/pulls/<n>/comments \
     --jq '.[] | select(.user.login | startswith("copilot")) | {path, line, body}'
   ```
   For each comment either **fix it** (commit + push, which triggers a fresh review)
   or **reply with the reason it does not apply** and resolve the thread. Resolving
   threads is done via GraphQL (`resolveReviewThread`) or in the web UI; never
   resolve a thread without a fix or a written justification.
4. Repeat 2–3 until CI is green **and** Copilot has reviewed the latest push with no
   unresolved threads.
5. Only then merge (`gh pr merge <n> --squash --auto` is fine). Never merge red, never
   merge without a Copilot review.
6. Do not use the admin bypass. If a human explicitly asks for it in an emergency,
   note it in the PR description.

Never push directly to `main`, never force-push a shared branch, never disable or
weaken a CI check to get green.

## Things that are easy to get wrong

- For d-cores, `DecompositionResult.node_indices` holds `max(k_in, k_out)` per node so
  the layout code works unchanged; the full `(k_in, k_out)` pairs live in
  `result.metadata["d_cores"]`. Writers branch on `result.decomp_type`.
- Layout is expensive on big graphs; tests must use tiny graphs (karate club is the
  upper bound).
- `examples/` download real CAIDA data over HTTPS; never run them in tests.
- `.python-version` pins 3.10 (the floor). Run `uv run --python 3.12 …` when you need a
  newer interpreter locally.
