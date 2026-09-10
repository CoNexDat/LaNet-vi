# Contributing to LaNet-vi

Thanks for your interest in LaNet-vi. This guide covers the development setup, the
checks every change must pass, and how a change gets from your machine into `main`.

## Development setup

LaNet-vi uses [uv](https://github.com/astral-sh/uv) for everything: environments,
dependencies, building, and publishing.

```bash
git clone git@github.com:CoNexDat/LaNet-vi.git
cd LaNet-vi
uv sync --all-extras          # creates .venv with runtime + dev + interactive extras
uv run pre-commit install     # installs the git hooks (ruff lint/format, file hygiene)
```

Run the same checks CI runs:

```bash
uv run ruff check src/ tests/ examples/
uv run ruff format --check src/ tests/ examples/
uv run mypy src/lanet_vi
uv run pytest                 # coverage threshold is enforced (see pyproject.toml)
uv build
```

`uv run pre-commit run --all-files` runs the lint and hygiene hooks on the whole tree.

## Project conventions

- Python 3.9+ compatible code (no `match`, no `X | Y` unions at runtime; ruff's `UP`
  rules are configured for the 3.9 target).
- Line length 100, ruff formatter, imports sorted by ruff (`I`).
- NumPy-style docstrings on public functions and classes (ruff `D` rules).
- Configuration objects are Pydantic models in `src/lanet_vi/models/config.py`. Add
  a field there and thread it through the CLI in `src/lanet_vi/cli.py` rather than
  adding ad-hoc parameters.
- Type hints everywhere; mypy runs with `disallow_untyped_defs`.
- Every behaviour change gets a test under `tests/` and a line in `CHANGELOG.md`
  under `[Unreleased]`.

## Making a change

`main` is protected. Nobody pushes to it directly; every change lands through a pull
request that has passed CI **and** a Copilot code review. This is repository policy: a PR
without a Copilot review is not merged.

1. Create a branch from `main`: `git switch -c <type>/<short-name>` (`feat/`, `fix/`,
   `docs/`, `chore/`).
2. Commit in small, coherent steps. The pre-commit hooks run ruff on each commit.
3. Push and open a PR: `gh pr create --fill`. The PR template has a checklist.
4. **Iterate until green.** Two things must be true before merging:
   - CI is green: `lint`, every `test (...)` matrix leg, and `build`.
   - A Copilot code review has run on the latest push and **every comment has been
     addressed**: either fix it and push, or reply explaining why it is not applicable
     and resolve the thread. Do not merge with unresolved review threads.

   The ruleset requests Copilot automatically on PRs targeting `main`. For stacked PRs
   or if no request appears, request it yourself and re-request it after each push
   (if Copilot shows "Monthly limit reached", wait for the quota to reset or upgrade;
   the PR stays open until then):

   ```bash
   gh pr edit <number> --add-reviewer Copilot   # or the Reviewers gear in the sidebar
   gh pr checks <number> --watch                # wait for CI
   gh pr view <number> --comments               # read review comments
   ```

5. Merge (squash or merge commit) once both are green. The branch is deleted
   automatically. Auto-merge is enabled on the repo, so `gh pr merge --auto --squash`
   is fine.

Repository admins can technically bypass the ruleset. Treat that as an emergency-only
escape hatch and say so in the PR when it is used.

## Releasing

1. Bump `version` in `pyproject.toml` (the only place the version lives;
   `lanet_vi.__version__` reads it from package metadata).
2. Move the `[Unreleased]` section of `CHANGELOG.md` under a new `[X.Y.Z] - YYYY-MM-DD`
   heading and add the compare link at the bottom.
3. Open a PR with those two changes and merge it.
4. Tag and publish a GitHub release: `git tag vX.Y.Z && git push origin vX.Y.Z`, then
   `gh release create vX.Y.Z --generate-notes`.
5. The **Publish to PyPI** workflow runs on the release and uploads with
   [trusted publishing](https://docs.pypi.org/trusted-publishers/) (OIDC, no token).

### One-time trusted-publishing setup

Trusted publishing must be configured once on PyPI before the workflow can upload:

- On PyPI, open the `lanet-vi` project → *Publishing* → *Add a new publisher* with
  owner `CoNexDat`, repository `LaNet-vi`, workflow `publish.yml`, environment `pypi`.
- On GitHub, create the `pypi` environment under *Settings → Environments* (optionally
  restrict it to tags matching `v*`).

Until this is done, the publish workflow will fail at the upload step.

## Reporting issues

Use the issue templates. For security problems follow `SECURITY.md` instead of opening
a public issue.
