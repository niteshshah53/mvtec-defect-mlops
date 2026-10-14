"""Unit tests for training configuration loading and CLI parsing.

No model training is performed inside unit tests.
"""

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from mvtec_defect.train import (
    TrainConfig,
    extract_metrics,
    load_params,
    main,
    parse_args,
    resolve_output_paths,
    train_model,
)


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


def test_load_params_valid_flat(tmp_path: Path) -> None:
    """Verify loading valid configuration with flat keys."""
    yaml_file = tmp_path / "params.yaml"
    _write_yaml(
        yaml_file,
        {
            "category": "bottle",
            "backbone": "wide_resnet50_2",
            "coreset_sampling_ratio": 0.1,
            "image_size": [256, 256],
            "seed": 42,
            "data_root": "data/mvtec_ad",
            "mlflow_tracking_uri": "sqlite:///mlflow.db",
            "mlflow_experiment_name": "mvtec_defect_patchcore",
        },
    )

    config = load_params(yaml_file)

    assert isinstance(config, TrainConfig)
    assert config.category == "bottle"
    assert config.backbone == "wide_resnet50_2"
    assert config.coreset_sampling_ratio == 0.1
    assert config.image_size == (256, 256)
    assert config.seed == 42
    assert config.test_split_mode == "from_dir"
    assert config.val_split_mode == "from_test"
    assert config.val_split_ratio == 0.5
    assert config.data_root == Path("data/mvtec_ad")
    assert config.mlflow_tracking_uri == "sqlite:///mlflow.db"
    assert config.mlflow_experiment_name == "mvtec_defect_patchcore"


def test_load_params_category_override(tmp_path: Path) -> None:
    """Verify CLI category override takes precedence over file configuration."""
    yaml_file = tmp_path / "params.yaml"
    _write_yaml(
        yaml_file,
        {
            "category": "bottle",
            "backbone": "wide_resnet50_2",
            "coreset_sampling_ratio": 0.1,
            "image_size": [256, 256],
            "seed": 42,
        },
    )

    config = load_params(yaml_file, category_override="screw")

    assert config.category == "screw"


def test_load_params_nested_train(tmp_path: Path) -> None:
    """Verify configuration nested under 'train' section is loaded."""
    yaml_file = tmp_path / "params.yaml"
    _write_yaml(
        yaml_file,
        {
            "train": {
                "category": "capsule",
                "backbone": "resnet18",
                "coreset_sampling_ratio": 0.05,
                "image_size": [224, 224],
                "seed": 123,
            }
        },
    )

    config = load_params(yaml_file)

    assert config.category == "capsule"
    assert config.backbone == "resnet18"
    assert config.coreset_sampling_ratio == 0.05
    assert config.image_size == (224, 224)
    assert config.seed == 123


def test_load_params_image_size_int(tmp_path: Path) -> None:
    """Verify single integer image_size is expanded to (size, size)."""
    yaml_file = tmp_path / "params.yaml"
    _write_yaml(
        yaml_file,
        {
            "category": "bottle",
            "backbone": "wide_resnet50_2",
            "coreset_sampling_ratio": 0.1,
            "image_size": 256,
            "seed": 42,
        },
    )

    config = load_params(yaml_file)
    assert config.image_size == (256, 256)


def test_load_params_file_not_found(tmp_path: Path) -> None:
    """Verify FileNotFoundError is raised if config file is missing."""
    missing = tmp_path / "nonexistent.yaml"
    with pytest.raises(FileNotFoundError, match="Configuration file does not exist"):
        load_params(missing)


def test_load_params_not_a_mapping(tmp_path: Path) -> None:
    """Verify ValueError is raised if YAML root is not a mapping."""
    yaml_file = tmp_path / "invalid.yaml"
    yaml_file.write_text("['not', 'a', 'dict']", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected YAML root to be a mapping"):
        load_params(yaml_file)


def test_load_params_missing_required_keys(tmp_path: Path) -> None:
    """Verify ValueError is raised if required keys are missing."""
    yaml_file = tmp_path / "params.yaml"
    _write_yaml(
        yaml_file,
        {
            "category": "bottle",
            "backbone": "wide_resnet50_2",
            # missing coreset_sampling_ratio, image_size, seed
        },
    )

    with pytest.raises(ValueError, match="Missing required configuration key"):
        load_params(yaml_file)


def test_load_params_invalid_category(tmp_path: Path) -> None:
    """Verify ValueError is raised if category is whitespace or empty."""
    yaml_file = tmp_path / "params.yaml"
    _write_yaml(
        yaml_file,
        {
            "category": "   ",
            "backbone": "wide_resnet50_2",
            "coreset_sampling_ratio": 0.1,
            "image_size": [256, 256],
            "seed": 42,
        },
    )

    with pytest.raises(ValueError, match="category must be a non-empty string"):
        load_params(yaml_file)


def test_load_params_invalid_coreset_ratio(tmp_path: Path) -> None:
    """Verify ValueError is raised if coreset_sampling_ratio is out of bounds."""
    yaml_file = tmp_path / "params.yaml"
    _write_yaml(
        yaml_file,
        {
            "category": "bottle",
            "backbone": "wide_resnet50_2",
            "coreset_sampling_ratio": 0.0,
            "image_size": [256, 256],
            "seed": 42,
        },
    )

    with pytest.raises(ValueError, match="coreset_sampling_ratio must be in"):
        load_params(yaml_file)

    _write_yaml(
        yaml_file,
        {
            "category": "bottle",
            "backbone": "wide_resnet50_2",
            "coreset_sampling_ratio": 1.5,
            "image_size": [256, 256],
            "seed": 42,
        },
    )

    with pytest.raises(ValueError, match="coreset_sampling_ratio must be in"):
        load_params(yaml_file)


def test_load_params_invalid_image_size(tmp_path: Path) -> None:
    """Verify ValueError is raised for malformed image_size."""
    yaml_file = tmp_path / "params.yaml"
    _write_yaml(
        yaml_file,
        {
            "category": "bottle",
            "backbone": "wide_resnet50_2",
            "coreset_sampling_ratio": 0.1,
            "image_size": [256, 256, 256],
            "seed": 42,
        },
    )

    with pytest.raises(ValueError, match="image_size list/tuple must have 2 elements"):
        load_params(yaml_file)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("test_split_mode", "same_as_test", "test_split_mode must be one of"),
        ("test_split_mode", "from_test", "test_split_mode must be one of"),
        ("test_split_mode", "unsupported", "test_split_mode must be one of"),
        ("val_split_mode", "same_as_test", "same_as_test.*not allowed"),
        ("val_split_mode", "unsupported", "val_split_mode must be one of"),
        ("val_split_ratio", "not-a-number", "val_split_ratio must be a numeric value"),
        ("val_split_ratio", 0.0, "val_split_ratio must be in"),
        ("val_split_ratio", 1.0, "val_split_ratio must be in"),
    ],
)
def test_load_params_invalid_split_settings(
    tmp_path: Path,
    key: str,
    value: object,
    match: str,
) -> None:
    """Verify split modes and validation ratio are validated."""
    yaml_file = tmp_path / "params.yaml"
    data = {
        "category": "bottle",
        "backbone": "wide_resnet50_2",
        "coreset_sampling_ratio": 0.1,
        "image_size": [256, 256],
        "seed": 42,
        key: value,
    }
    _write_yaml(yaml_file, data)

    with pytest.raises(ValueError, match=match):
        load_params(yaml_file)


def test_train_config_to_dict() -> None:
    """Verify TrainConfig.to_dict formats parameters cleanly for MLflow."""
    config = TrainConfig(
        category="bottle",
        backbone="wide_resnet50_2",
        coreset_sampling_ratio=0.1,
        image_size=(256, 256),
        seed=42,
        test_split_mode="from_dir",
        val_split_mode="from_test",
        val_split_ratio=0.5,
        data_root=Path("data/mvtec_ad"),
    )

    params = config.to_dict()
    assert params["category"] == "bottle"
    assert params["backbone"] == "wide_resnet50_2"
    assert params["coreset_sampling_ratio"] == 0.1
    assert params["image_size"] == "256x256"
    assert params["seed"] == 42
    assert params["test_split_mode"] == "from_dir"
    assert params["val_split_mode"] == "from_test"
    assert params["val_split_ratio"] == 0.5
    assert params["data_root"] == "data/mvtec_ad"


def test_parse_args_defaults() -> None:
    """Verify default CLI arguments."""
    args = parse_args([])
    assert args.category is None
    assert args.params == Path("params.yaml")
    assert args.data_root is None
    assert args.output_dir is None
    assert args.metrics_file is None


def test_parse_args_explicit() -> None:
    """Verify explicit CLI arguments."""
    args = parse_args(
        [
            "--category",
            "screw",
            "--params",
            "custom_params.yaml",
            "--data-root",
            "custom/data",
            "--output-dir",
            "custom/artifacts",
            "--metrics-file",
            "custom/metrics.json",
        ]
    )
    assert args.category == "screw"
    assert args.params == Path("custom_params.yaml")
    assert args.data_root == Path("custom/data")
    assert args.output_dir == Path("custom/artifacts")
    assert args.metrics_file == Path("custom/metrics.json")


def test_resolve_output_paths_defaults() -> None:
    """Verify historical output paths remain the defaults."""
    output_dir, metrics_file = resolve_output_paths("bottle")

    assert output_dir == Path("artifacts/bottle")
    assert metrics_file == Path("metrics/bottle.json")


def test_resolve_output_paths_explicit() -> None:
    """Verify explicit output paths are used without category suffixes."""
    output_dir, metrics_file = resolve_output_paths(
        "bottle",
        output_dir="runs/week2/bottle",
        metrics_file="runs/week2/metrics.json",
    )

    assert output_dir == Path("runs/week2/bottle")
    assert metrics_file == Path("runs/week2/metrics.json")


def test_extract_metrics() -> None:
    """Verify extract_metrics correctly extracts float metrics."""

    class FakeTensor:
        def __init__(self, val: float):
            self.val = val

        def item(self) -> float:
            return self.val

    raw_results = [
        {
            "image_AUROC": FakeTensor(1.0),
            "image_F1Score": 0.9920,
            "pixel_AUROC": 0.9856,
            "non_numeric": "ignored",
        }
    ]

    metrics = extract_metrics(raw_results)
    assert metrics == {
        "image_AUROC": 1.0,
        "image_F1Score": 0.9920,
        "pixel_AUROC": 0.9856,
    }


@pytest.fixture
def mock_ml_deps() -> dict[str, Any]:
    """Create isolated mocks for lazy ML dependencies."""
    mock_mlflow = MagicMock()
    mock_run = MagicMock()
    mock_run.info.run_id = "test_run_123"
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_run
    mock_cm.__exit__.return_value = False
    mock_mlflow.start_run.return_value = mock_cm

    mock_data = MagicMock()
    mock_split = MagicMock()

    mock_engine_module = MagicMock()
    mock_engine = MagicMock()
    mock_engine.test.return_value = [{"image_AUROC": 0.992, "pixel_AUROC": 0.985}]
    mock_engine.best_model_path = None
    mock_engine_module.Engine.return_value = mock_engine

    mock_models = MagicMock()
    mock_patchcore_cls = MagicMock()
    mock_models.Patchcore = mock_patchcore_cls
    mock_model_instance = MagicMock()
    mock_patchcore_cls.return_value = mock_model_instance
    mock_patchcore_cls.configure_pre_processor.return_value = MagicMock()

    mock_lightning = MagicMock()

    modules = {
        "mlflow": mock_mlflow,
        "anomalib.data": mock_data,
        "anomalib.data.utils.split": mock_split,
        "anomalib.engine": mock_engine_module,
        "anomalib.models": mock_models,
        "lightning.pytorch": mock_lightning,
    }
    return {
        "modules": modules,
        "mlflow": mock_mlflow,
        "start_run_cm": mock_cm,
        "run": mock_run,
        "engine": mock_engine,
        "engine_module": mock_engine_module,
        "models": mock_models,
        "patchcore_cls": mock_patchcore_cls,
        "data": mock_data,
        "lightning": mock_lightning,
    }


def test_train_model_success_lifecycle(
    tmp_path: Path, mock_ml_deps: dict[str, Any]
) -> None:
    """Verify MLflow setup, parameter logging, training, evaluation, and artifacts."""
    data_root = tmp_path / "data"
    (data_root / "bottle").mkdir(parents=True)
    output_dir = tmp_path / "artifacts" / "bottle"
    metrics_file = tmp_path / "metrics" / "bottle.json"

    # Pre-create a checkpoint to test artifact logging
    ckpt_dir = output_dir / "weights" / "lightning"
    ckpt_dir.mkdir(parents=True)
    ckpt_file = ckpt_dir / "model.ckpt"
    ckpt_file.write_text("fake checkpoint", encoding="utf-8")

    config = TrainConfig(
        category="bottle",
        backbone="wide_resnet50_2",
        coreset_sampling_ratio=0.1,
        image_size=(256, 256),
        seed=42,
        data_root=data_root,
        mlflow_tracking_uri="sqlite:///test_mlflow.db",
        mlflow_experiment_name="test_experiment",
    )

    with patch.dict(sys.modules, mock_ml_deps["modules"]):
        metrics = train_model(
            config,
            output_dir=output_dir,
            metrics_file=metrics_file,
        )

    # 1. Returned metrics match extract_metrics
    assert metrics == {"image_AUROC": 0.992, "pixel_AUROC": 0.985}

    # 2. MLflow configured
    mock_mlflow = mock_ml_deps["mlflow"]
    mock_mlflow.set_tracking_uri.assert_called_once_with("sqlite:///test_mlflow.db")
    mock_mlflow.set_experiment.assert_called_once_with("test_experiment")

    # 3. Descriptive run name with category, backbone, and coreset ratio
    mock_mlflow.start_run.assert_called_once()
    run_name = mock_mlflow.start_run.call_args.kwargs.get("run_name", "")
    assert "bottle" in run_name
    assert "wide_resnet50_2" in run_name
    assert "0.1" in run_name
    assert run_name == "bottle_wide_resnet50_2_0.1"

    # 4. Parameters logged before training
    mock_mlflow.log_params.assert_called_once_with(config.to_dict())

    # 5. Training and evaluation executed
    mock_ml_deps["engine"].train.assert_called_once()
    mock_ml_deps["engine"].test.assert_called_once()

    # 6. Only actual metrics logged to MLflow
    mock_mlflow.log_metrics.assert_called_once_with(
        {"image_AUROC": 0.992, "pixel_AUROC": 0.985}
    )

    # 7. Elapsed duration logged in seconds
    mock_mlflow.log_metric.assert_called_once()
    metric_key, metric_val = mock_mlflow.log_metric.call_args[0]
    assert metric_key == "duration_seconds"
    assert isinstance(metric_val, float)
    assert metric_val >= 0.0

    # 8. Checkpoint and metrics artifacts logged
    artifact_calls = [call[0] for call in mock_mlflow.log_artifact.call_args_list]
    logged_paths = [str(call[0]) for call in artifact_calls]
    assert str(ckpt_file) in logged_paths
    assert str(metrics_file) in logged_paths

    # 9. Metrics JSON file saved
    assert metrics_file.is_file()
    saved_metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert saved_metrics == {"image_AUROC": 0.992, "pixel_AUROC": 0.985}

    # 10. Verify run context manager received clean exit without exceptions
    mock_cm = mock_ml_deps["start_run_cm"]
    mock_cm.__exit__.assert_called_once_with(None, None, None)


def test_train_model_params_logged_before_training(
    tmp_path: Path, mock_ml_deps: dict[str, Any]
) -> None:
    """Verify parameters are logged before model training begins."""
    data_root = tmp_path / "data"
    (data_root / "bottle").mkdir(parents=True)

    config = TrainConfig(category="bottle", data_root=data_root)

    call_order: list[str] = []
    mock_ml_deps["mlflow"].log_params.side_effect = lambda *_args, **_kwargs: (
        call_order.append("log_params")
    )
    mock_ml_deps["engine"].train.side_effect = lambda *_args, **_kwargs: (
        call_order.append("engine_train")
    )
    mock_ml_deps["engine"].test.side_effect = lambda *_args, **_kwargs: (
        call_order.append("engine_test")
    )

    with patch.dict(sys.modules, mock_ml_deps["modules"]):
        train_model(
            config, output_dir=tmp_path / "out", metrics_file=tmp_path / "m.json"
        )

    assert call_order == ["log_params", "engine_train", "engine_test"]


def test_train_model_failure_during_training(
    tmp_path: Path, mock_ml_deps: dict[str, Any]
) -> None:
    """Verify training failure propagates, records duration, and skips artifacts."""
    data_root = tmp_path / "data"
    (data_root / "bottle").mkdir(parents=True)
    metrics_file = tmp_path / "metrics.json"

    config = TrainConfig(category="bottle", data_root=data_root)
    mock_ml_deps["engine"].train.side_effect = RuntimeError("CUDA out of memory")

    with (
        patch.dict(sys.modules, mock_ml_deps["modules"]),
        pytest.raises(RuntimeError, match="CUDA out of memory"),
    ):
        train_model(config, output_dir=tmp_path / "out", metrics_file=metrics_file)

    mock_mlflow = mock_ml_deps["mlflow"]
    # Run was started
    mock_mlflow.start_run.assert_called_once()
    # Parameters were logged before failure
    mock_mlflow.log_params.assert_called_once_with(config.to_dict())
    # Duration was recorded in finally block despite the exception
    mock_mlflow.log_metric.assert_called_once()
    metric_key, metric_val = mock_mlflow.log_metric.call_args[0]
    assert metric_key == "duration_seconds"
    assert isinstance(metric_val, float)
    assert metric_val >= 0.0

    # No evaluation metrics or artifacts logged
    mock_mlflow.log_metrics.assert_not_called()
    mock_mlflow.log_artifact.assert_not_called()
    assert not metrics_file.exists()

    # Verify context manager received the exception escaping the run context
    # so real MLflow context manager can mark the run as failed.
    mock_cm = mock_ml_deps["start_run_cm"]
    mock_cm.__exit__.assert_called_once()
    exc_type, exc_val, _ = mock_cm.__exit__.call_args[0]
    assert exc_type is RuntimeError
    assert "CUDA out of memory" in str(exc_val)


def test_train_model_failure_during_evaluation(
    tmp_path: Path, mock_ml_deps: dict[str, Any]
) -> None:
    """Verify eval failure propagates, records duration, and skips artifacts."""
    data_root = tmp_path / "data"
    (data_root / "bottle").mkdir(parents=True)
    metrics_file = tmp_path / "metrics.json"

    config = TrainConfig(category="bottle", data_root=data_root)
    mock_ml_deps["engine"].test.side_effect = RuntimeError("Evaluation failed")

    with (
        patch.dict(sys.modules, mock_ml_deps["modules"]),
        pytest.raises(RuntimeError, match="Evaluation failed"),
    ):
        train_model(config, output_dir=tmp_path / "out", metrics_file=metrics_file)

    mock_mlflow = mock_ml_deps["mlflow"]
    mock_mlflow.start_run.assert_called_once()
    mock_ml_deps["engine"].train.assert_called_once()
    # Duration recorded despite failure
    mock_mlflow.log_metric.assert_called_once()
    metric_key, metric_val = mock_mlflow.log_metric.call_args[0]
    assert metric_key == "duration_seconds"
    assert isinstance(metric_val, float)
    assert metric_val >= 0.0

    # No evaluation metrics or artifacts logged
    mock_mlflow.log_metrics.assert_not_called()
    mock_mlflow.log_artifact.assert_not_called()
    assert not metrics_file.exists()

    # Verify context manager received evaluation exception
    mock_cm = mock_ml_deps["start_run_cm"]
    mock_cm.__exit__.assert_called_once()
    exc_type, exc_val, _ = mock_cm.__exit__.call_args[0]
    assert exc_type is RuntimeError
    assert "Evaluation failed" in str(exc_val)


def test_train_model_failure_during_model_setup(
    tmp_path: Path, mock_ml_deps: dict[str, Any]
) -> None:
    """Verify model setup failure inside run context propagates and logs duration."""
    data_root = tmp_path / "data"
    (data_root / "bottle").mkdir(parents=True)

    config = TrainConfig(category="bottle", data_root=data_root)
    mock_ml_deps["patchcore_cls"].side_effect = ValueError(
        "Invalid backbone architecture"
    )

    with (
        patch.dict(sys.modules, mock_ml_deps["modules"]),
        pytest.raises(ValueError, match="Invalid backbone architecture"),
    ):
        train_model(
            config, output_dir=tmp_path / "out", metrics_file=tmp_path / "m.json"
        )

    mock_mlflow = mock_ml_deps["mlflow"]
    mock_mlflow.start_run.assert_called_once()
    mock_ml_deps["engine"].train.assert_not_called()
    mock_mlflow.log_metric.assert_called_once()
    metric_key, metric_val = mock_mlflow.log_metric.call_args[0]
    assert metric_key == "duration_seconds"

    # Verify context manager received model setup exception
    mock_cm = mock_ml_deps["start_run_cm"]
    mock_cm.__exit__.assert_called_once()
    exc_type, exc_val, _ = mock_cm.__exit__.call_args[0]
    assert exc_type is ValueError
    assert "Invalid backbone architecture" in str(exc_val)


def test_train_model_duration_logging_failure_preserves_original_exception(
    tmp_path: Path, mock_ml_deps: dict[str, Any]
) -> None:
    """Verify failure in duration logging does not mask original training error."""
    data_root = tmp_path / "data"
    (data_root / "bottle").mkdir(parents=True)

    config = TrainConfig(category="bottle", data_root=data_root)
    mock_ml_deps["engine"].train.side_effect = RuntimeError("Original training failure")
    mock_ml_deps["mlflow"].log_metric.side_effect = RuntimeError(
        "Duration metric logging failed"
    )

    with (
        patch.dict(sys.modules, mock_ml_deps["modules"]),
        pytest.raises(RuntimeError, match="Original training failure"),
    ):
        train_model(
            config, output_dir=tmp_path / "out", metrics_file=tmp_path / "m.json"
        )

    # Verify context manager received original training exception, not duration error
    mock_cm = mock_ml_deps["start_run_cm"]
    mock_cm.__exit__.assert_called_once()
    exc_type, exc_val, _ = mock_cm.__exit__.call_args[0]
    assert exc_type is RuntimeError
    assert "Original training failure" in str(exc_val)


def test_train_model_duration_logging_failure_without_prior_error(
    tmp_path: Path, mock_ml_deps: dict[str, Any]
) -> None:
    """Verify duration logging error on success propagates to fail the run."""
    data_root = tmp_path / "data"
    (data_root / "bottle").mkdir(parents=True)

    config = TrainConfig(category="bottle", data_root=data_root)
    mock_ml_deps["mlflow"].log_metric.side_effect = RuntimeError(
        "Duration metric failure on success"
    )

    with (
        patch.dict(sys.modules, mock_ml_deps["modules"]),
        pytest.raises(RuntimeError, match="Duration metric failure on success"),
    ):
        train_model(
            config, output_dir=tmp_path / "out", metrics_file=tmp_path / "m.json"
        )

    # Verify the context manager received the metric logging exception
    mock_cm = mock_ml_deps["start_run_cm"]
    mock_cm.__exit__.assert_called_once()
    exc_type, exc_val, _ = mock_cm.__exit__.call_args[0]
    assert exc_type is RuntimeError
    assert "Duration metric failure on success" in str(exc_val)


def test_train_model_missing_category_directory(
    tmp_path: Path, mock_ml_deps: dict[str, Any]
) -> None:
    """Verify FileNotFoundError is raised before MLflow run when dataset is missing."""
    config = TrainConfig(category="nonexistent_category", data_root=tmp_path)

    with (
        patch.dict(sys.modules, mock_ml_deps["modules"]),
        pytest.raises(FileNotFoundError, match="Category directory does not exist"),
    ):
        train_model(config)

    mock_ml_deps["mlflow"].start_run.assert_not_called()


def test_main_cli_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CLI main() returns exit code 0 on successful training."""
    yaml_file = tmp_path / "params.yaml"
    _write_yaml(
        yaml_file,
        {
            "category": "bottle",
            "backbone": "wide_resnet50_2",
            "coreset_sampling_ratio": 0.1,
            "image_size": [256, 256],
            "seed": 42,
            "data_root": str(tmp_path),
        },
    )

    monkeypatch.setattr(
        "mvtec_defect.train.train_model",
        lambda *_args, **_kwargs: {"image_AUROC": 0.99},
    )

    exit_code = main(["--params", str(yaml_file), "--category", "bottle"])
    assert exit_code == 0


def test_main_cli_failure_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify CLI main() catches training exceptions and returns exit code 1."""
    yaml_file = tmp_path / "params.yaml"
    _write_yaml(
        yaml_file,
        {
            "category": "bottle",
            "backbone": "wide_resnet50_2",
            "coreset_sampling_ratio": 0.1,
            "image_size": [256, 256],
            "seed": 42,
            "data_root": str(tmp_path),
        },
    )

    def _failing_train(*_args: Any, **_kwargs: Any) -> dict[str, float]:
        raise RuntimeError("Simulated GPU crash")

    monkeypatch.setattr("mvtec_defect.train.train_model", _failing_train)

    exit_code = main(["--params", str(yaml_file), "--category", "bottle"])
    assert exit_code == 1
