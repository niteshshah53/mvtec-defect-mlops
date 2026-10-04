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

## 4. GitHub Actions CI Pipeline
- **Date**: 2026-10-04
- **Decision**: Use `astral-sh/setup-uv@v5` with caching enabled, `uv python install 3.11`, and `uv sync --locked` to run Ruff (lint + format check) and pytest on pushes to `main` and pull requests.
- **Rationale**: Keeps CI fast, deterministic, and isolated. Avoids installing heavy ML dependencies (Anomalib, PyTorch) or downloading dataset files in the PR/linting loop.

## 5. MVTec AD Dataset Validation and Summary Schema
- **Date**: 2026-10-04
- **Decision**: Implement dataset structure validation and counting via `summarize_dataset(root, categories)` using standard library modules only (`pathlib`, `json`, `argparse`). Validation errors raise `DatasetValidationError`, inheriting from both `ValueError` and `FileNotFoundError`. Validates folder structure (`train/good`, `test/good`, `test/<defect>`, `ground_truth/<defect>`) and image-to-mask count equality for every defect type. Returns a plain dictionary mapping category names to summary metrics (`train_good`, `test_good`, `test_defects`, `test_defects_total`, `test_total`, `ground_truth_masks`, `ground_truth_defects`).
- **Rationale**: Ensures reproducible and automated data validation across categories (bottle, screw, capsule) before training, while keeping CI and tests lightweight with zero heavy runtime dependencies.

## 6. Dataset Inspection Image Library
- **Date**: 2026-10-04
- **Decision**: Use the already-declared Pillow dependency for dataset contact-sheet generation.
- **Rationale**: Pillow provides the required image loading, resizing, compositing, labeling, and PNG output without adding matplotlib, numpy, or other dependencies.

## 7. Contact-Sheet Visual Inspection
- **Date**: 2026-10-04
- **Observation**: The inspected contact sheets showed substantial variation in defect size, location, and shape, including large regions, thin line-like defects, and multi-component masks. Several defects were subtle in raw images, especially some screw and capsule examples. Good samples also varied in screw orientation, capsule imprint/logo visibility, and bottle appearance/reflections.
