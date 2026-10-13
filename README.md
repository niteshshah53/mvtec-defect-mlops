# MVTec Defect MLOps

An anomaly-detection portfolio project for industrial visual inspection using
the MVTec AD dataset. The project focuses on a reproducible workflow for
validating image data, training and evaluating an Anomalib PatchCore model,
tracking experiments locally, and managing pipeline outputs with DVC.

## Motivation

Industrial inspection systems need more than a model: they also need
repeatable data checks, explicit configuration, traceable metrics, and
artifacts that can be regenerated. This repository demonstrates those
foundations for the MVTec AD `bottle`, `screw`, and `capsule` categories.

## Technology

- Python 3.11
- PyTorch
- Anomalib 2.7.0 with PatchCore
- DVC
- MLflow 3.17.0
- uv
- pytest and Ruff for testing and code quality

## Pipeline overview

```text
MVTec AD dataset
        |
        v
Dataset validation and summary
        |
        v
PatchCore training and evaluation
        |
        +--> metrics/<category>.json
        +--> artifacts/<category>/
        +--> local MLflow run and model/metric artifacts
```

The DVC pipeline defines two stages:

1. `prepare` validates the configured category and writes
   `reports/data_summary.json`.
2. `train` consumes the validated dataset and summary, trains and evaluates
   PatchCore, and writes the category-specific model artifacts and metrics.

The current DVC configuration uses the category in `params.yaml`, which is
currently `bottle`.

## Prerequisites and setup

Install [uv](https://docs.astral.sh/uv/) and obtain Python 3.11. The repository
pins the supported Python range to `>=3.11,<3.12` and records `3.11` in
`.python-version`.

From the repository root:

```powershell
uv python install 3.11
uv sync --locked
```

The project specifies CUDA 13.0 PyTorch wheels through the configured uv
package index. Confirm that the resulting environment is appropriate for the
machine where training will run.

## Dataset

The MVTec AD dataset is **not included** in this repository and must be
obtained separately. Place it under `data/mvtec_ad/` with the following
structure for each category:

```text
data/mvtec_ad/
└── <category>/
    ├── train/
    │   └── good/
    ├── test/
    │   ├── good/
    │   └── <defect>/
    └── ground_truth/
        └── <defect>/
```

The validator checks that the required directories contain images and that
each test defect has a matching set of ground-truth masks.

## Configuration

The active configuration is in `params.yaml`:

```yaml
category: bottle
backbone: wide_resnet50_2
coreset_sampling_ratio: 0.1
image_size: [256, 256]
seed: 42
data_root: data/mvtec_ad
```

It also contains the local MLflow tracking URI and experiment name. The
training CLI loads this file with `--params params.yaml`.

## Usage

### Validate the configured dataset

For the current `bottle` configuration:

```powershell
uv run python -m mvtec_defect.data --root data/mvtec_ad --categories bottle --output reports/data_summary.json
```

### Reproduce the DVC pipeline

```powershell
uv run dvc repro
```

The current pipeline is configured for the category selected in `params.yaml`.

### View metrics

The repository contains category metric files that can be displayed through
DVC:

```powershell
uv run dvc metrics show
uv run dvc metrics diff
```

The training code writes metrics as JSON under `metrics/<category>.json`.

### View local MLflow tracking

Training uses the local SQLite tracking URI `sqlite:///mlflow.db` and the
experiment name `mvtec_defect_patchcore`. After a run, start the MLflow UI
against that store:

```powershell
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

The training code also logs the run parameters, returned metrics, the model
checkpoint when available, and the metrics JSON as an MLflow artifact.

### Test, lint, and format

Run the repository checks directly:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The Makefile provides equivalent targets:

```powershell
make test
make lint
make format
```

These targets invoke `uv run` internally. Use the direct commands above if
`make` is not installed.

## Baseline results

These values are read from the existing JSON files in `metrics/` and rounded
to four decimal places. They are reported here as measured repository
artifacts, not as guarantees for other datasets or configurations.

| Category | Image AUROC | Image F1 | Pixel AUROC | Pixel F1 |
| --- | ---: | ---: | ---: | ---: |
| Bottle | 1.0000 | 0.9920 | 0.9857 | 0.7273 |
| Screw | 0.9643 | 0.9544 | 0.9890 | 0.3736 |
| Capsule | 0.9932 | 0.9772 | 0.9900 | 0.5154 |

Image-level metrics evaluate whether an image is identified as anomalous.
Pixel-level metrics evaluate the localization of anomalous regions against
ground-truth masks. The two metric groups measure different aspects of the
task and should not be treated as interchangeable.

## Reproducibility and tracking

The training configuration is loaded from `params.yaml`, including the
category, PatchCore backbone, coreset sampling ratio, image size, random seed,
and dataset root. DVC records the dataset and source dependencies, the
validation summary, category-specific artifacts, metrics, and declared
parameters in the pipeline.

MLflow is configured for a local SQLite store at `mlflow.db`. Runs use the
`mvtec_defect_patchcore` experiment and record the resolved training
parameters, evaluation metrics, and available model and metrics artifacts.
The local tracking database, downloaded data, and model weights are not
intended for source control.

## Limitations

- Baseline results are currently available only for `bottle`, `screw`, and
  `capsule`.
- The current DVC pipeline is configured for the category in `params.yaml`,
  which is `bottle`; screw and capsule are not reproduced by the current
  pipeline definition.
- Metrics depend on the dataset contents and the configuration used for the
  run.
- This is a portfolio prototype, not a production-ready inspection system.
- Deployment, online serving, and production monitoring are outside the
  current implementation.
