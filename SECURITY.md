# Security policy

## Supported versions

Security fixes are released for the latest minor version.

| Version | Supported |
|---|---|
| 0.2.x | ✅ |
| < 0.2 | ❌ |

## Formula safety

Decision files may come from untrusted sources, so formulas never reach `eval` or `exec`. They are parsed with `ast` and only whitelisted nodes are compiled (see `src/mchermes/formula.py`). Any way to execute code, read files or access attributes through a formula, a YAML file or a `given()` condition is a security bug.

## Reporting a vulnerability

Please **do not open a public issue**. Use GitHub's
[private vulnerability reporting](https://github.com/tetttet/montecarlo-hermes/security/advisories/new)
with a description and, ideally, a minimal reproduction. You will get a response within
7 days, and a fix or mitigation plan as soon as the issue is confirmed.
