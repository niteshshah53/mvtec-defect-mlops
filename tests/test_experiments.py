"""Unit tests for the PatchCore experiment matrix runner and preflight protection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from mvtec_defect.experiments import (
    BACKBONES,
    CATEGORIES,
    CORESET_RATIOS,
    DEFAULT_ARTIFACTS_ROOT,
    DEFAULT_METRICS_ROOT,
    ExperimentSpec,
    PreflightError,
    build_experiment_specs,
    format_ratio,
    get_experiment_paths,
    main,
    run_experiments,
    run_preflight_checks,
    select_pilot_spec,
    validate_cli_selections,
)
from mvtec_defect.train import TrainConfig


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


@pytest.fixture
def base_config(tmp_path: Path) -> TrainConfig:
    """Create a standard base TrainConfig pointing to tmp_path."""
    return TrainConfig(
        category="bottle",
        backbone="wide_resnet50_2",
        coreset_sampling_ratio=0.1,
        image_size=(256, 256),
        seed=42,
        test_split_mode="from_dir",
        val_split_mode="from_test",
        val_split_ratio=0.5,
        data_root=tmp_path / "data",
        mlflow_tracking_uri="sqlite:///test.db",
        mlflow_experiment_name="test_exp",
    )


def test_build_experiment_specs_matrix_count_and_uniqueness(
    base_config: TrainConfig,
) -> None:
    """Verify matrix generates exactly 12 unique configurations with unique paths."""
    specs = build_experiment_specs(base_config)

    assert len(specs) == 12

    # Check configuration uniqueness
    config_tuples = [(s.category, s.backbone, s.coreset_sampling_ratio) for s in specs]
    assert len(set(config_tuples)) == 12

    # Check output directory uniqueness
    output_dirs = [s.output_dir for s in specs]
    assert len(set(output_dirs)) == 12

    # Check metrics file uniqueness
    metrics_files = [s.metrics_file for s in specs]
    assert len(set(metrics_files)) == 12

    # Check domain values
    assert {s.category for s in specs} == set(CATEGORIES)
    assert {s.backbone for s in specs} == set(BACKBONES)
    assert {s.coreset_sampling_ratio for s in specs} == set(CORESET_RATIOS)


def test_build_experiment_specs_preserves_base_settings(tmp_path: Path) -> None:
    """Verify base settings (seed, splits, image size, data/MLflow) are preserved."""
    custom_base = TrainConfig(
        category="bottle",
        backbone="wide_resnet50_2",
        coreset_sampling_ratio=0.1,
        image_size=(256, 256),
        seed=999,
        test_split_mode="from_dir",
        val_split_mode="from_test",
        val_split_ratio=0.5,
        data_root=tmp_path / "custom_data",
        mlflow_tracking_uri="sqlite:///custom.db",
        mlflow_experiment_name="custom_experiment",
    )

    specs = build_experiment_specs(custom_base)

    for spec in specs:
        cfg = spec.config
        assert cfg.seed == 999
        assert cfg.image_size == (256, 256)
        assert cfg.test_split_mode == "from_dir"
        assert cfg.val_split_mode == "from_test"
        assert cfg.val_split_ratio == 0.5
        assert cfg.data_root == tmp_path / "custom_data"
        assert cfg.mlflow_tracking_uri == "sqlite:///custom.db"
        assert cfg.mlflow_experiment_name == "custom_experiment"


def test_format_ratio_normalization() -> None:
    """Verify ratio formatting normalizes trailing zeros (0.1 vs 0.10)."""
    assert format_ratio(0.1) == "0.1"
    assert format_ratio("0.1") == "0.1"
    assert format_ratio(0.10) == "0.1"
    assert format_ratio("0.10") == "0.1"
    assert format_ratio(0.01) == "0.01"
    assert format_ratio("0.01") == "0.01"
    assert format_ratio(0.010) == "0.01"


def test_get_experiment_paths_isolation() -> None:
    """Verify output paths are isolated under day3 directories with canonical naming."""
    out_dir, metrics_file = get_experiment_paths("bottle", "wide_resnet50_2", 0.01)

    assert out_dir == DEFAULT_ARTIFACTS_ROOT / "bottle" / "wide_resnet50_2_ratio_0.01"
    assert (
        metrics_file
        == DEFAULT_METRICS_ROOT / "bottle" / "wide_resnet50_2_ratio_0.01.json"
    )

    # Ensure no collision with baseline/historical outputs
    assert out_dir != Path("artifacts/bottle")
    assert out_dir != Path("artifacts/experiments/bottle")
    assert metrics_file != Path("metrics/bottle.json")
    assert metrics_file != Path("metrics/experiments/bottle.json")


def test_select_pilot_spec(base_config: TrainConfig) -> None:
    """Verify pilot selection isolates bottle + wide_resnet50_2 + ratio 0.01."""
    specs = build_experiment_specs(base_config)
    pilot = select_pilot_spec(specs)

    assert len(pilot) == 1
    spec = pilot[0]
    assert spec.category == "bottle"
    assert spec.backbone == "wide_resnet50_2"
    assert spec.coreset_sampling_ratio == 0.01


def test_select_pilot_spec_missing_raises() -> None:
    """Verify ValueError is raised if pilot spec is not found in provided list."""
    empty_specs: list[ExperimentSpec] = []
    with pytest.raises(ValueError, match="Pilot configuration .* not found"):
        select_pilot_spec(empty_specs)


def test_preflight_blocks_existing_artifact_directory(
    base_config: TrainConfig, tmp_path: Path
) -> None:
    """Verify preflight aborts when an artifact destination directory exists."""
    art_root = tmp_path / "artifacts"
    metrics_root = tmp_path / "metrics"
    specs = build_experiment_specs(
        base_config, artifacts_root=art_root, metrics_root=metrics_root
    )

    # Pre-create the artifact directory for one spec
    specs[0].output_dir.mkdir(parents=True)

    with pytest.raises(PreflightError, match="Intended destination.*already exist"):
        run_preflight_checks(specs)


def test_preflight_blocks_existing_metrics_file(
    base_config: TrainConfig, tmp_path: Path
) -> None:
    """Verify preflight aborts when an intended metrics file exists."""
    art_root = tmp_path / "artifacts"
    metrics_root = tmp_path / "metrics"
    specs = build_experiment_specs(
        base_config, artifacts_root=art_root, metrics_root=metrics_root
    )

    # Pre-create a metrics file
    specs[2].metrics_file.parent.mkdir(parents=True, exist_ok=True)
    specs[2].metrics_file.write_text("{}", encoding="utf-8")

    with pytest.raises(PreflightError, match="Intended destination.*already exist"):
        run_preflight_checks(specs)


def test_preflight_rejects_duplicate_destinations(base_config: TrainConfig) -> None:
    """Verify preflight rejects duplicate destination paths in the batch."""
    spec1 = ExperimentSpec(
        category="bottle",
        backbone="wide_resnet50_2",
        coreset_sampling_ratio=0.01,
        config=base_config,
        output_dir=Path("artifacts/shared"),
        metrics_file=Path("metrics/spec1.json"),
    )
    spec2 = ExperimentSpec(
        category="screw",
        backbone="resnet50",
        coreset_sampling_ratio=0.1,
        config=base_config,
        output_dir=Path("artifacts/shared"),  # Duplicate!
        metrics_file=Path("metrics/spec2.json"),
    )

    with pytest.raises(PreflightError, match="Duplicate artifact destination"):
        run_preflight_checks([spec1, spec2])


def test_preflight_empty_specs_rejected() -> None:
    """Verify preflight rejects an empty specification list."""
    with pytest.raises(PreflightError, match="No experiment configurations"):
        run_preflight_checks([])


def test_preflight_blocks_execution_before_any_training(
    base_config: TrainConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify preflight failure prevents train_model from ever being called."""
    art_root = tmp_path / "artifacts"
    metrics_root = tmp_path / "metrics"
    specs = build_experiment_specs(
        base_config, artifacts_root=art_root, metrics_root=metrics_root
    )

    # Mark the LAST spec destination as existing
    specs[-1].metrics_file.parent.mkdir(parents=True, exist_ok=True)
    specs[-1].metrics_file.write_text("{}", encoding="utf-8")

    mock_train = MagicMock()
    monkeypatch.setattr("mvtec_defect.experiments.train_model", mock_train)

    with pytest.raises(PreflightError):
        run_experiments(specs, dry_run=False)

    # Crucial: train_model was never invoked, not even for specs[0]
    mock_train.assert_not_called()


def test_dry_run_never_calls_train_model(
    base_config: TrainConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify dry_run executes listing without calling train_model."""
    specs = build_experiment_specs(base_config)
    mock_train = MagicMock()
    monkeypatch.setattr("mvtec_defect.experiments.train_model", mock_train)

    results = run_experiments(specs, dry_run=True)

    assert results == []
    mock_train.assert_not_called()


def test_successful_execution_passes_isolated_paths(
    base_config: TrainConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify successful run passes isolated paths to train_model and records result."""
    art_root = tmp_path / "artifacts"
    metrics_root = tmp_path / "metrics"
    specs = build_experiment_specs(
        base_config, artifacts_root=art_root, metrics_root=metrics_root
    )
    pilot = select_pilot_spec(specs)

    mock_train = MagicMock(return_value={"image_AUROC": 0.995})
    monkeypatch.setattr("mvtec_defect.experiments.train_model", mock_train)

    results = run_experiments(pilot, dry_run=False)

    assert len(results) == 1
    assert results[0].status == "success"
    assert results[0].metrics == {"image_AUROC": 0.995}
    assert results[0].error is None

    mock_train.assert_called_once_with(
        config=pilot[0].config,
        output_dir=pilot[0].output_dir,
        metrics_file=pilot[0].metrics_file,
    )


def test_training_failure_stops_batch(
    base_config: TrainConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify batch stops on the first failure and records error status."""
    art_root = tmp_path / "artifacts"
    metrics_root = tmp_path / "metrics"
    specs = build_experiment_specs(
        base_config, artifacts_root=art_root, metrics_root=metrics_root
    )
    # Take first two specs
    test_specs = specs[:2]

    # First call fails
    mock_train = MagicMock(side_effect=RuntimeError("Simulated CUDA failure"))
    monkeypatch.setattr("mvtec_defect.experiments.train_model", mock_train)

    results = run_experiments(test_specs, dry_run=False)

    # Stopped on first failure
    assert len(results) == 1
    assert results[0].status == "failed"
    assert results[0].metrics is None
    assert "Simulated CUDA failure" in str(results[0].error)
    assert mock_train.call_count == 1


def test_main_cli_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CLI main with --dry-run prints plan and returns exit code 0."""
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

    mock_train = MagicMock()
    monkeypatch.setattr("mvtec_defect.experiments.train_model", mock_train)

    exit_code = main(["--params", str(yaml_file), "--dry-run"])
    assert exit_code == 0
    mock_train.assert_not_called()


def test_main_cli_pilot_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify CLI main with --dry-run --pilot lists only pilot and returns 0."""
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

    mock_train = MagicMock()
    monkeypatch.setattr("mvtec_defect.experiments.train_model", mock_train)

    exit_code = main(["--params", str(yaml_file), "--dry-run", "--pilot"])
    assert exit_code == 0
    mock_train.assert_not_called()


def test_main_cli_pilot_execution_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify CLI main --pilot runs successfully and returns 0."""
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

    # Redirect default paths to tmp_path
    monkeypatch.setattr(
        "mvtec_defect.experiments.DEFAULT_ARTIFACTS_ROOT", tmp_path / "art"
    )
    monkeypatch.setattr(
        "mvtec_defect.experiments.DEFAULT_METRICS_ROOT", tmp_path / "met"
    )

    mock_train = MagicMock(return_value={"image_AUROC": 0.99})
    monkeypatch.setattr("mvtec_defect.experiments.train_model", mock_train)

    exit_code = main(["--params", str(yaml_file), "--pilot"])
    assert exit_code == 0
    mock_train.assert_called_once()


def test_main_cli_preflight_failure_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify CLI main returns 1 when preflight detects existing destination."""
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

    art_dir = tmp_path / "art" / "bottle" / "wide_resnet50_2_ratio_0.01"
    art_dir.mkdir(parents=True)

    monkeypatch.setattr(
        "mvtec_defect.experiments.DEFAULT_ARTIFACTS_ROOT", tmp_path / "art"
    )
    monkeypatch.setattr(
        "mvtec_defect.experiments.DEFAULT_METRICS_ROOT", tmp_path / "met"
    )

    mock_train = MagicMock()
    monkeypatch.setattr("mvtec_defect.experiments.train_model", mock_train)

    exit_code = main(["--params", str(yaml_file), "--pilot"])
    assert exit_code == 1
    mock_train.assert_not_called()


def test_main_cli_training_failure_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify CLI main returns 1 on training failure."""
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
        "mvtec_defect.experiments.DEFAULT_ARTIFACTS_ROOT", tmp_path / "art"
    )
    monkeypatch.setattr(
        "mvtec_defect.experiments.DEFAULT_METRICS_ROOT", tmp_path / "met"
    )

    mock_train = MagicMock(side_effect=RuntimeError("GPU OOM"))
    monkeypatch.setattr("mvtec_defect.experiments.train_model", mock_train)

    exit_code = main(["--params", str(yaml_file), "--pilot"])
    assert exit_code == 1


def test_validate_cli_selections_valid() -> None:
    """Verify validate_cli_selections accepts valid matrix combinations and None."""
    validate_cli_selections()
    validate_cli_selections(category="bottle", backbone="wide_resnet50_2", ratio=0.01)
    validate_cli_selections(category="screw", backbone="resnet50", ratio=0.1)
    validate_cli_selections(category="capsule", backbone="wide_resnet50_2", ratio=0.10)


def test_validate_cli_selections_invalid_category() -> None:
    """Verify invalid category raises ValueError."""
    with pytest.raises(ValueError, match="Invalid category 'metal_nut'"):
        validate_cli_selections(category="metal_nut")


def test_validate_cli_selections_invalid_backbone() -> None:
    """Verify invalid backbone raises ValueError."""
    with pytest.raises(ValueError, match="Invalid backbone 'resnet18'"):
        validate_cli_selections(backbone="resnet18")


def test_validate_cli_selections_invalid_ratio() -> None:
    """Verify invalid ratio raises ValueError."""
    with pytest.raises(ValueError, match="Invalid ratio '0.5'"):
        validate_cli_selections(ratio=0.5)


def test_main_cli_rejects_invalid_category(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify CLI rejects invalid category and never calls train_model."""
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
    mock_train = MagicMock()
    monkeypatch.setattr("mvtec_defect.experiments.train_model", mock_train)

    with caplog.at_level("ERROR"):
        exit_code = main(["--params", str(yaml_file), "--category", "invalid_category"])

    assert exit_code != 0
    assert "Invalid category 'invalid_category'" in caplog.text
    mock_train.assert_not_called()


def test_main_cli_rejects_invalid_backbone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify CLI rejects invalid backbone and never calls train_model."""
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
    mock_train = MagicMock()
    monkeypatch.setattr("mvtec_defect.experiments.train_model", mock_train)

    with caplog.at_level("ERROR"):
        exit_code = main(["--params", str(yaml_file), "--backbone", "invalid_backbone"])

    assert exit_code != 0
    assert "Invalid backbone 'invalid_backbone'" in caplog.text
    mock_train.assert_not_called()


def test_main_cli_rejects_invalid_ratio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify CLI rejects invalid ratio and never calls train_model."""
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
    mock_train = MagicMock()
    monkeypatch.setattr("mvtec_defect.experiments.train_model", mock_train)

    with caplog.at_level("ERROR"):
        exit_code = main(["--params", str(yaml_file), "--ratio", "0.5"])

    assert exit_code != 0
    assert "Invalid ratio '0.5'" in caplog.text
    mock_train.assert_not_called()


def test_build_experiment_specs_rejects_invalid_domain() -> None:
    """Verify build_experiment_specs rejects invalid domain configurations."""
    cfg = TrainConfig(category="bottle")
    with pytest.raises(ValueError, match="Invalid category 'unsupported'"):
        build_experiment_specs(cfg, categories=["unsupported"])
    with pytest.raises(ValueError, match="Invalid backbone 'unsupported'"):
        build_experiment_specs(cfg, backbones=["unsupported"])
    with pytest.raises(ValueError, match="Invalid ratio '0.5'"):
        build_experiment_specs(cfg, ratios=[0.5])
