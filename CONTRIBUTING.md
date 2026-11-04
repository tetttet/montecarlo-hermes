# Contributing

Thanks for helping make decision-making under uncertainty easier! Bug reports, new
examples, distributions and docs improvements are all welcome.

## Development setup

```bash
git clone https://github.com/tetttet/montecarlo-hermes.git
cd montecarlo-hermes
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before you open a pull request

```bash
ruff check src tests examples scripts
ruff format src tests examples scripts
mypy
pytest            # fails below 90% coverage
```

CI runs the same checks on Python 3.10, 3.11 and 3.12.

## Guidelines

- **Keep the public API tiny.** New features should feel like `pandas`: one obvious
  call with sensible defaults. Open an issue first for anything that adds API surface.
- **Stay vectorized.** Formulas run on whole sample arrays; avoid Python loops over
  samples.
- **Stay reproducible.** The same seed must give identical results. Each variable
  draws from its own stream (`mchermes.decision.stream`).
- **Never `eval` user text.** Formula features go through the `ast` whitelist in
  `mchermes/formula.py`, with tests that prove what is rejected.
- **Type and document everything.** Public functions get type hints and docstrings;
  `mypy --strict` must pass.
- **Charts.** If you change plotting, regenerate the README images with
  `python scripts/generate_assets.py` and include them in the PR.

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`,
`docs:`, `test:`, `ci:`, `chore:`, `refactor:`.

## Releasing (maintainers)

1. Update the version in `pyproject.toml` and `src/mchermes/__init__.py`, and the
   changelog.
2. Tag and push: `git tag v0.x.y && git push origin v0.x.y`. The `Publish to PyPI`
   workflow builds and uploads with trusted publishing.
