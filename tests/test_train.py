"""Unit tests for training configuration loading and CLI parsing.

No model training is performed inside unit tests.
"""

from pathlib import Path

import pytest
import yaml

from mvtec_defect.train import (
    TrainConfig,
    extract_metrics,
    load_params,
    parse_args,
    resolve_output_paths,
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
