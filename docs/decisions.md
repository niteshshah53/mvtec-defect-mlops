# Architecture and Tooling Decisions

## 1. Project Scaffolding and Build Backend
- **Date**: 2026-10-04
- **Decision**: Use `hatchling` as the build backend with a standard `src/` layout (`src/mvtec_defect/`).
- **Rationale**: `hatchling` is the default and best-supported PEP 517/621 build backend for `uv`. It cleanly handles src-based packaging and editable installs.

## 2. Python Version Pinning
- **Date**: 2026-10-04
- **Decision**: Target Python 3.11 via `requires-python = ">=3.11,<3.12"` in `pyproject.toml` and `.python-version`.
- **Rationale**: Strict compatibility with Anomalib and PyTorch stacks specified in project requirements.

## 3. Dependency Management
- **Date**: 2026-10-04
- **Decision**: Manage development dependencies (`pytest`, `ruff`, `pre-commit`) via PEP 735 `[dependency-groups].dev`.
- **Rationale**: Keeps runtime dependencies decoupled from developer and CI tooling while integrating seamlessly with `uv sync`.
