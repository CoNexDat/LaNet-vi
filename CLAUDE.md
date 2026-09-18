@AGENTS.md

## Claude Code way of working (overrides the Copilot part of AGENTS.md)

The Copilot review rule in AGENTS.md is kept for other contributors; in Claude Code
sessions **do not request or wait for Copilot** (paid quota). Instead, every PR is
reviewed by a **fresh-context reviewer subagent**:

1. Push the branch, open the PR, wait for CI (`gh pr checks <n> --watch`).
2. With the PR branch checked out, spawn a **new** reviewer with the Agent tool:
   `model: "sonnet"`, `isolation: "worktree"`, self-contained prompt — repository and PR
   number, `gh pr diff <n>`, the C++ files under `legacy/` the change ports (if any), the
   conventions in AGENTS.md, the checks to run (`uv run pytest -q`, ruff, mypy),
   "read-only, do not modify the repository", and the required output: findings ranked
   by severity with `file:line`, a concrete failing scenario and a suggested fix, ending
   with exactly `VERDICT: APPROVE` or `VERDICT: CHANGES REQUESTED`.
3. Read every finding; verify it before acting. Fix it (commit + push) or record in a PR
   comment why it does not apply.
4. After any change, spawn **another fresh** reviewer for the new state — never continue
   the previous reviewer session.
5. Repeat until CI is green on the latest push **and** the latest reviewer says
   `VERDICT: APPROVE`. Post a PR comment summarizing the rounds (what was fixed, what was
   declined and why), then merge with `gh pr merge <n> --squash --delete-branch`.

Everything else in AGENTS.md (branching, CI, never merge red, no admin bypass) applies.
