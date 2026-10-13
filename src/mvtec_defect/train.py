"""MVTec AD Patchcore model training and evaluation pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Configuration for Patchcore training on MVTec AD."""

    category: str
    backbone: str = "wide_resnet50_2"
    coreset_sampling_ratio: float = 0.1
    image_size: tuple[int, int] = (256, 256)
    seed: int = 42
    data_root: Path = Path("data/mvtec_ad")
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "mvtec_defect_patchcore"

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary suitable for MLflow parameter logging."""
        data = asdict(self)
        data["image_size"] = f"{self.image_size[0]}x{self.image_size[1]}"
        data["data_root"] = self.data_root.as_posix()
        return data


def resolve_output_paths(
    category: str,
    output_dir: Path | str | None = None,
    metrics_file: Path | str | None = None,
) -> tuple[Path, Path]:
    """Resolve training artifact and metrics output paths.

    When overrides are omitted, preserve the historical category-based paths.
    """
    resolved_output_dir = (
        Path(output_dir) if output_dir is not None else Path("artifacts") / category
    )
    resolved_metrics_file = (
        Path(metrics_file)
        if metrics_file is not None
        else Path("metrics") / f"{category}.json"
    )
    return resolved_output_dir, resolved_metrics_file


def _parse_image_size(raw_size: Any) -> tuple[int, int]:
    """Parse image size from int or pair of ints."""
    if isinstance(raw_size, int):
        if raw_size <= 0:
            msg = f"image_size dimension must be positive, got {raw_size}"
            raise ValueError(msg)
        return (raw_size, raw_size)
    if isinstance(raw_size, (list, tuple)):
        if len(raw_size) != 2:
            msg = f"image_size list/tuple must have 2 elements, got {len(raw_size)}"
            raise ValueError(msg)
        h, w = int(raw_size[0]), int(raw_size[1])
        if h <= 0 or w <= 0:
            msg = f"image_size dimensions must be positive, got ({h}, {w})"
            raise ValueError(msg)
        return (h, w)
    msg = f"image_size must be an integer or a pair of integers, got {raw_size!r}"
    raise ValueError(msg)


def load_params(
    params_path: Path | str = "params.yaml",
    category_override: str | None = None,
) -> TrainConfig:
    """Load and validate training configuration from a YAML file.

    Args:
        params_path: Path to params.yaml.
        category_override: Optional category override from CLI.

    Returns:
        Validated TrainConfig instance.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If required fields are missing or invalid.
    """
    path = Path(params_path)
    if not path.is_file():
        msg = f"Configuration file does not exist: {path}"
        raise FileNotFoundError(msg)

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        msg = f"Expected YAML root to be a mapping, got {type(data).__name__}"
        raise ValueError(msg)

    # Allow configuration nested under 'train' or at top-level
    if "train" in data and isinstance(data["train"], dict):
        cfg_dict: dict[str, Any] = data["train"]
    else:
        cfg_dict = data

    required_keys = {
        "category",
        "backbone",
        "coreset_sampling_ratio",
        "image_size",
        "seed",
    }
    missing = required_keys - set(cfg_dict.keys())
    if missing:
        msg = f"Missing required configuration key(s) in {path}: {sorted(missing)}"
        raise ValueError(msg)

    raw_category = (
        category_override if category_override is not None else cfg_dict["category"]
    )
    category = str(raw_category).strip()
    if not category:
        msg = "category must be a non-empty string"
        raise ValueError(msg)

    backbone = str(cfg_dict["backbone"]).strip()
    if not backbone:
        msg = "backbone must be a non-empty string"
        raise ValueError(msg)

    try:
        coreset_ratio = float(cfg_dict["coreset_sampling_ratio"])
    except (TypeError, ValueError) as err:
        msg = "coreset_sampling_ratio must be a numeric value"
        raise ValueError(msg) from err

    if not (0.0 < coreset_ratio <= 1.0):
        msg = f"coreset_sampling_ratio must be in (0.0, 1.0], got {coreset_ratio}"
        raise ValueError(msg)

    image_size = _parse_image_size(cfg_dict["image_size"])

    try:
        seed = int(cfg_dict["seed"])
    except (TypeError, ValueError) as err:
        msg = "seed must be an integer"
        raise ValueError(msg) from err

    data_root = Path(cfg_dict.get("data_root", "data/mvtec_ad"))
    mlflow_tracking_uri = str(
        cfg_dict.get("mlflow_tracking_uri", "sqlite:///mlflow.db")
    )
    mlflow_experiment_name = str(
        cfg_dict.get("mlflow_experiment_name", "mvtec_defect_patchcore")
    )

    return TrainConfig(
        category=category,
        backbone=backbone,
        coreset_sampling_ratio=coreset_ratio,
        image_size=image_size,
        seed=seed,
        data_root=data_root,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for training.

    Args:
        args: Command-line arguments.

    Returns:
        Parsed arguments Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train Anomalib PatchCore model on MVTec AD category.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help=(
            "MVTec AD category to train (e.g., bottle, screw, capsule). "
            "Overrides params.yaml."
        ),
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=Path("params.yaml"),
        help="Path to params.yaml configuration file (default: params.yaml).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root path to MVTec AD dataset directory. Overrides params.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=("Directory for training artifacts. Defaults to artifacts/<category>."),
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=None,
        help="Path for the metrics JSON file. Defaults to metrics/<category>.json.",
    )
    return parser.parse_args(args)


def extract_metrics(test_results: Any) -> dict[str, float]:
    """Extract and serialize metrics returned by Anomalib Engine.test.

    Args:
        test_results: Output from Engine.test (list of mappings or mapping).

    Returns:
        Dictionary mapping metric names to native float values.
    """
    metrics: dict[str, float] = {}

    def _process_mapping(mapping: Mapping[Any, Any]) -> None:
        for k, v in mapping.items():
            if hasattr(v, "item"):
                v = v.item()
            try:
                metrics[str(k)] = float(v)
            except (TypeError, ValueError):
                continue

    if isinstance(test_results, list):
        for entry in test_results:
            if isinstance(entry, Mapping):
                _process_mapping(entry)
    elif isinstance(test_results, Mapping):
        _process_mapping(test_results)

    return metrics


def train_model(
    config: TrainConfig,
    output_dir: Path | str | None = None,
    metrics_file: Path | str | None = None,
) -> dict[str, float]:
    """Train and test PatchCore model on the specified category.

    Args:
        config: Verified training configuration.
        output_dir: Optional directory for training artifacts.
        metrics_file: Optional path for the metrics JSON file.

    Returns:
        Dictionary of actual test metrics returned by Engine.test.

    Raises:
        FileNotFoundError: If the category dataset folder does not exist.
    """
    # Lazy imports to keep lightweight test runner and CI fast
    import mlflow
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
    from lightning.pytorch import seed_everything

    category_dir = config.data_root / config.category
    if not category_dir.is_dir():
        msg = f"Category directory does not exist: {category_dir}"
        raise FileNotFoundError(msg)

    # 1. Seed experiment
    logger.info("Setting global seed to %d", config.seed)
    seed_everything(config.seed, workers=True)

    # 2. Setup DataModule
    logger.info("Initializing MVTecAD dataset for category: %s", config.category)
    datamodule = MVTecAD(
        root=config.data_root,
        category=config.category,
        seed=config.seed,
    )

    # 3. Setup PreProcessor and Model
    logger.info(
        "Configuring PatchCore with backbone=%s, ratio=%.3f, image_size=%s",
        config.backbone,
        config.coreset_sampling_ratio,
        config.image_size,
    )
    pre_processor = Patchcore.configure_pre_processor(image_size=config.image_size)
    model = Patchcore(
        backbone=config.backbone,
        coreset_sampling_ratio=config.coreset_sampling_ratio,
        pre_processor=pre_processor,
    )

    # 4. Setup Engine
    output_dir, metrics_path = resolve_output_paths(
        category=config.category,
        output_dir=output_dir,
        metrics_file=metrics_file,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = Engine(
        default_root_dir=output_dir,
    )

    # 5. Train model
    logger.info("Starting PatchCore training...")
    engine.train(model=model, datamodule=datamodule)

    # 6. Test model
    logger.info("Running evaluation on test set...")
    test_results = engine.test(model=model, datamodule=datamodule)

    # 7. Extract actual metrics returned by Engine
    metrics = extract_metrics(test_results)
    print("\n--- Actual Test Metrics ---")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v}")

    # 8. Save metrics to the configured output path
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(f"\nMetrics written to: {metrics_path}")

    # 9. Log parameters, actual metrics, and model artifact to MLflow
    logger.info("Logging run to MLflow at %s", config.mlflow_tracking_uri)
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    with mlflow.start_run(run_name=f"{config.category}_patchcore") as run:
        mlflow.log_params(config.to_dict())
        if metrics:
            mlflow.log_metrics(metrics)

        checkpoint_path = engine.best_model_path
        if not checkpoint_path or not Path(checkpoint_path).exists():
            # Check standard fallback path
            fallback_ckpt = output_dir / "weights" / "lightning" / "model.ckpt"
            if fallback_ckpt.exists():
                checkpoint_path = str(fallback_ckpt)

        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"Logging model checkpoint artifact to MLflow: {checkpoint_path}")
            mlflow.log_artifact(str(checkpoint_path), artifact_path="model")
        else:
            print("Warning: No model checkpoint found to log to MLflow.")

        if metrics_path.exists():
            mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

        print(f"MLflow Run ID: {run.info.run_id}")

    return metrics


def main(argv: list[str] | None = None) -> int:
    """Execute PatchCore training CLI.

    Args:
        argv: Command-line argument vector.

    Returns:
        0 on success, non-zero on error.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)

    try:
        config = load_params(
            params_path=args.params,
            category_override=args.category,
        )
        if args.data_root is not None:
            config.data_root = args.data_root

        train_model(
            config,
            output_dir=args.output_dir,
            metrics_file=args.metrics_file,
        )
    except Exception as err:
        logger.error("Training failed: %s", err, exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
