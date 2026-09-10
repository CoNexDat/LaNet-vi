# Security Policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| 5.x     | Yes       |
| < 5.0   | No (legacy C++ releases) |

## Reporting a vulnerability

Please do **not** open a public issue for security problems.

Report privately through GitHub Security Advisories:
<https://github.com/CoNexDat/LaNet-vi/security/advisories/new>

Include the version, a description of the issue, and a minimal reproduction if you have one.
You will get an acknowledgement within a week. Fixes are released as a patch version and
noted in `CHANGELOG.md`.

LaNet-vi reads edge-list files and downloads public datasets over HTTPS. It does not run
untrusted code and has no network-facing service, so the main risk area is parsing
malformed input files.
