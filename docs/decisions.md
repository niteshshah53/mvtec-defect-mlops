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

## 8. Anomalib PatchCore Training and MLflow Tracking Pipeline
- **Date**: 2026-10-13
- **Decision**: Implement `src/mvtec_defect/train.py` using Anomalib 2.7.0 (`MVTecAD`, `Patchcore`, `Engine`), Lightning seeding (`seed_everything`), and MLflow (`sqlite:///mlflow.db`) experiment tracking.
- **Key Details**:
  1. Configuration: Loaded and validated from `params.yaml` with explicit settings for category, backbone, coreset sampling ratio, image size, and seed. The CLI `--category` argument allows overriding the active category while preserving other defaults.
  2. Pre-processing: Uses `Patchcore.configure_pre_processor(image_size=config.image_size)` to construct standard resize and normalization transformations.
  3. Dynamic Metric Extraction: Metric dictionaries returned by `Engine.test()` are dynamically extracted and serialized without hardcoded metric names or values, saved to `metrics/<category>.json`.
  4. MLflow Tracking: Tracks runs under experiment `mvtec_defect_patchcore` with SQLite backend `sqlite:///mlflow.db`. Logs hyperparameters, actual test metrics, the saved model checkpoint artifact, and the metrics file.
  5. Lazy ML Imports: Heavy ML imports (`anomalib`, `torch`, `lightning`, `mlflow`) are deferred to runtime inside `train_model()` so parameter validation, CLI parsing, and unit tests execute instantaneously in lightweight CI environments (`uv sync --locked`) without requiring heavy GPU/PyTorch packages.

## 9. Disjoint Validation and Test Evaluation Protocol
- **Date**: 2026-10-13
- **Decision**: Use the explicit protocol `test_split_mode=from_dir`, `val_split_mode=from_test`, and `val_split_ratio=0.5` with seed 42.
- **Rationale**: Anomalib derives score-normalization statistics and F1 decision thresholds from validation outputs, not from test outputs. The revised protocol makes validation and test image paths disjoint while retaining both image labels for the supported categories.
- **Limitation**: Validation samples are split from the official test directory, but the resulting validation and test subsets are disjoint. This changes the evaluated test subset and must be disclosed when comparing results with the frozen Week 1 baseline.
