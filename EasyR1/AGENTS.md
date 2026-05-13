# Repository Guidelines

## Project Structure & Module Organization
- `verl/` holds the core Python package (training engine, algorithms, data, runtime).
- `configs/repro/` contains the retained SeePhys Pro reproduction configs.
- `examples/repro/` holds the retained PhysRL / visual-math normal-vs-blind launch scripts.

## Build, Test, and Development Commands
- `pip install -e .` installs the package in editable mode for local development.
- `make build` builds source and wheel distributions.
- `make quality` runs Ruff lint and format checks.
- `make style` auto-fixes lint issues and formats code with Ruff.
- `make commit` installs and runs pre-commit hooks.

## Coding Style & Naming Conventions
- Python 3.9+, 4-space indentation, and a 119-character line length.
- Ruff is the formatter/linter; it enforces double quotes and isort-style imports.
- Follow existing naming: modules/functions `snake_case`, classes `CamelCase`, constants `UPPER_SNAKE_CASE`.
- Reproduction scripts encode domain, model size, and control condition, e.g.
  `examples/repro/physrl_4b_blind.sh`.

## Commit & Pull Request Guidelines
- Git history shows short summaries like "Update ..." and occasional scoped tags like `[algorithm]`.
- Prefer concise, imperative summaries; add a bracketed scope when touching algorithms or infra.
- Avoid placeholder messages like `1`.
- PRs should describe the change, link issues, list commands run, and include new or updated
  `examples/` or `configs/` when changing training behavior.
