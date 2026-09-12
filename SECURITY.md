# Security Policy

## Supported versions

Only the latest 5.x release on PyPI receives security fixes.

| Version | Supported |
| ------- | --------- |
| 5.x (latest release) | Yes |
| 5.x (older releases) | No — upgrade to the latest 5.x |
| < 5.0 (legacy C++)   | No |

LaNet-vi 5.x requires Python 3.10 or newer. Python 3.9 reached end of life in October
2025 and several dependencies (including Pillow) no longer ship security fixes for it, so
it is not supported.

## Reporting a vulnerability

Please do **not** open a public issue, pull request or discussion for security problems.

Report privately through GitHub's private vulnerability reporting for this repository:
<https://github.com/CoNexDat/LaNet-vi/security/advisories/new>

You need a GitHub account. If that is not possible, contact the maintainer listed in
[`CODEOWNERS`](.github/CODEOWNERS) through a private channel and ask for one.

Please include:

- The LaNet-vi version (`lanet-vi version` or `pip show lanet-vi`) and Python version.
- What the issue is and what an attacker can do with it.
- A minimal reproduction: an input file, a command line or a short script.
- Whether you have already disclosed it anywhere else.

## What to expect

The project is maintained by researchers on a volunteer basis, so response times are
best-effort:

| Step | Target |
| ---- | ------ |
| Acknowledgement of your report | within 7 days |
| Initial assessment (confirmed / not a vulnerability / needs more information) | within 14 days |
| Fix released for a confirmed issue | within 90 days, sooner for anything severe |

We follow coordinated disclosure: we will keep the report private while working on a
fix, credit you in the advisory and `CHANGELOG.md` unless you prefer otherwise, and
publish the GitHub Security Advisory when the fixed version is on PyPI. If we have not
released a fix 90 days after acknowledging a confirmed report, you are free to disclose it.

Fixes are shipped as a patch release of the current 5.x line.

## Scope

LaNet-vi is a library and command-line tool. It reads edge-list and configuration files,
optionally downloads public datasets over HTTPS, and writes images and data files. It has
no network-facing service and does not execute code from its inputs.

In scope:

- Anything that lets a crafted input file (edge list, YAML configuration, CAIDA snapshot)
  execute code, read or write files outside the requested output, or crash the process in
  a way that is more than a normal parse error.
- Unsafe defaults in the CLI or public Python API.
- Vulnerabilities in the GitHub Actions workflows or the PyPI release process.

Out of scope:

- Denial of service by simply providing a very large graph; the tool is designed for
  large inputs and memory use scales with them.
- Vulnerabilities in dependencies that do not affect how LaNet-vi uses them. We still
  update them — see below — but please report those upstream.

## Dependencies and automation

- Dependabot alerts and Dependabot security updates are enabled; security fixes for
  dependencies arrive as pull requests against `uv.lock`.
- The `Security` workflow runs [`pip-audit`](https://github.com/pypa/pip-audit) over the
  locked dependency tree (all extras) on every lockfile change and weekly, and
  [`zizmor`](https://github.com/zizmorcore/zizmor) over the workflow files.
- GitHub Actions are pinned to commit SHAs; the release workflow uses PyPI trusted
  publishing (OIDC) so no long-lived PyPI token exists.
- Secret scanning is enabled on the repository.
