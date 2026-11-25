# Security Policy

## Supported versions

Security fixes go into the latest released version of StatArb.jl. Older releases
are not patched; upgrade with `Pkg.update("StatArb")`.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | ✅        |

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub:
**Security → Report a vulnerability** on
[the repository page](https://github.com/rainerrodrigues/StatArb.jl/security/advisories/new).
Don't open a public issue for a security problem.

Include the StatArb and Julia versions, what an attacker could do, and a minimal
way to reproduce it. You can expect an acknowledgement within a week. If the
report is confirmed, a fix and an advisory will be published together, and you'll
be credited unless you'd rather not be.

## Scope

StatArb is a research backtester: it reads market data you supply (including
memory-mapped Arrow files) and never places real orders. Relevant reports include
crashes or memory-safety problems triggered by crafted input files, and issues in
the CI workflows. Trading losses from using its results are not security issues.
