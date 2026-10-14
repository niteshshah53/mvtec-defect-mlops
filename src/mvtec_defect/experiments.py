"""MVTec AD Patchcore experiment matrix runner with preflight protection."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from mvtec_defect.train import TrainConfig, load_params, train_model

logger = logging.getLogger(__name__)

CATEGORIES: tuple[str, ...] = ("bottle", "screw", "capsule")
BACKBONES: tuple[str, ...] = ("wide_resnet50_2", "resnet50")
CORESET_RATIOS: tuple[float, ...] = (0.01, 0.1)

DEFAULT_ARTIFACTS_ROOT = Path("artifacts/day3_experiments")
DEFAULT_METRICS_ROOT = Path("metrics/experiments/day3")

PILOT_CATEGORY = "bottle"
PILOT_BACKBONE = "wide_resnet50_2"
PILOT_RATIO = 0.01


class PreflightError(RuntimeError):
    """Raised when preflight safety checks fail before model training."""


@dataclass(frozen=True)
class ExperimentSpec:
    """Specification for an isolated experiment run in the matrix."""

    category: str
    backbone: str
    coreset_sampling_ratio: float
    config: TrainConfig
    output_dir: Path
    metrics_file: Path


@dataclass
class ExperimentResult:
    """Execution outcome for a single experiment run."""

    spec: ExperimentSpec
    status: str
    metrics: dict[str, float] | None = None
    error: str | None = None


def format_ratio(ratio: float | str) -> str:
    """Format coreset ratio consistently to prevent path divergence.

    Args:
        ratio: Numeric coreset sampling ratio (e.g. 0.1, 0.10, 0.01).

    Returns:
        Canonical string representation (e.g. "0.1", "0.01").
    """
    val = float(ratio)
    return f"{val:g}"


def get_experiment_paths(
    category: str,
    backbone: str,
    ratio: float,
    artifacts_root: Path | None = None,
    metrics_root: Path | None = None,
) -> tuple[Path, Path]:
    """Resolve isolated artifact and metrics paths for an experiment configuration.

    Args:
        category: MVTec AD category (e.g., bottle).
        backbone: Feature extractor backbone name.
        ratio: Coreset sampling ratio.
        artifacts_root: Root directory for day3 experiment artifacts.
            Defaults to DEFAULT_ARTIFACTS_ROOT at call time if None.
        metrics_root: Root directory for day3 experiment metrics.
            Defaults to DEFAULT_METRICS_ROOT at call time if None.

    Returns:
        Tuple of (output_dir, metrics_file).
    """
    if artifacts_root is None:
        artifacts_root = DEFAULT_ARTIFACTS_ROOT
    if metrics_root is None:
        metrics_root = DEFAULT_METRICS_ROOT
    norm_ratio = format_ratio(ratio)
    exp_name = f"{backbone}_ratio_{norm_ratio}"
    output_dir = artifacts_root / category / exp_name
    metrics_file = metrics_root / category / f"{exp_name}.json"
    return output_dir, metrics_file


def build_experiment_specs(
    base_config: TrainConfig,
    categories: Sequence[str] = CATEGORIES,
    backbones: Sequence[str] = BACKBONES,
    ratios: Sequence[float] = CORESET_RATIOS,
    artifacts_root: Path | None = None,
    metrics_root: Path | None = None,
) -> list[ExperimentSpec]:
    """Construct isolated experiment specifications from base configuration.

    Preserves base parameters (image_size, seed, split settings, data_root, MLflow)
    and parameterizes category, backbone, and coreset ratio in memory.

    Args:
        base_config: Base TrainConfig loaded from params.yaml.
        categories: Sequence of categories to include.
        backbones: Sequence of backbones to include.
        ratios: Sequence of coreset ratios to include.
        artifacts_root: Destination root directory for artifacts.
            Defaults to DEFAULT_ARTIFACTS_ROOT at call time if None.
        metrics_root: Destination root directory for metrics.
            Defaults to DEFAULT_METRICS_ROOT at call time if None.

    Returns:
        List of ExperimentSpec instances.
    """
    if artifacts_root is None:
        artifacts_root = DEFAULT_ARTIFACTS_ROOT
    if metrics_root is None:
        metrics_root = DEFAULT_METRICS_ROOT

    for cat in categories:
        if cat not in CATEGORIES:
            msg = (
                f"Invalid category '{cat}'. "
                f"Supported categories are: {', '.join(CATEGORIES)}."
            )
            raise ValueError(msg)
    for bb in backbones:
        if bb not in BACKBONES:
            msg = (
                f"Invalid backbone '{bb}'. "
                f"Supported backbones are: {', '.join(BACKBONES)}."
            )
            raise ValueError(msg)
    for ratio in ratios:
        if not any(math.isclose(float(ratio), r, abs_tol=1e-5) for r in CORESET_RATIOS):
            msg = (
                f"Invalid ratio '{ratio}'. "
                f"Supported ratios are: {', '.join(str(r) for r in CORESET_RATIOS)}."
            )
            raise ValueError(msg)

    specs: list[ExperimentSpec] = []
    for cat in categories:
        for bb in backbones:
            for ratio in ratios:
                norm_ratio = float(ratio)
                output_dir, metrics_file = get_experiment_paths(
                    category=cat,
                    backbone=bb,
                    ratio=norm_ratio,
                    artifacts_root=artifacts_root,
                    metrics_root=metrics_root,
                )
                run_config = dataclasses.replace(
                    base_config,
                    category=cat,
                    backbone=bb,
                    coreset_sampling_ratio=norm_ratio,
                )
                specs.append(
                    ExperimentSpec(
                        category=cat,
                        backbone=bb,
                        coreset_sampling_ratio=norm_ratio,
                        config=run_config,
                        output_dir=output_dir,
                        metrics_file=metrics_file,
                    )
                )
    return specs


def select_pilot_spec(
    specs: Sequence[ExperimentSpec],
    category: str = PILOT_CATEGORY,
    backbone: str = PILOT_BACKBONE,
    ratio: float = PILOT_RATIO,
) -> list[ExperimentSpec]:
    """Filter experiment specifications down to the single pilot run.

    Args:
        specs: Sequence of experiment specifications.
        category: Target pilot category (default: bottle).
        backbone: Target pilot backbone (default: wide_resnet50_2).
        ratio: Target pilot ratio (default: 0.01).

    Returns:
        List containing the single matching ExperimentSpec.

    Raises:
        ValueError: If matching pilot spec is not found.
    """
    for spec in specs:
        if (
            spec.category == category
            and spec.backbone == backbone
            and math.isclose(spec.coreset_sampling_ratio, ratio, abs_tol=1e-5)
        ):
            return [spec]
    msg = (
        f"Pilot configuration ({category}, {backbone}, {format_ratio(ratio)}) "
        "not found in experiment specifications."
    )
    raise ValueError(msg)


def run_preflight_checks(specs: Sequence[ExperimentSpec]) -> None:
    """Validate destination isolation and ensure no intended destinations exist.

    Args:
        specs: Sequence of planned experiment specifications.

    Raises:
        PreflightError: If specs is empty, destinations are duplicated, or any
            intended destination already exists on disk.
    """
    if not specs:
        msg = "Preflight check failed: No experiment configurations provided."
        raise PreflightError(msg)

    # 1. Check duplicate destinations
    seen_output_dirs: dict[Path, str] = {}
    seen_metrics_files: dict[Path, str] = {}
    for spec in specs:
        desc = (
            f"{spec.category}/{spec.backbone}/ratio_"
            f"{format_ratio(spec.coreset_sampling_ratio)}"
        )
        res_output = spec.output_dir.resolve()
        res_metrics = spec.metrics_file.resolve()

        if res_output in seen_output_dirs:
            msg = (
                f"Preflight check failed: Duplicate artifact destination detected.\n"
                f"Path: {spec.output_dir}\n"
                f"Config 1: {seen_output_dirs[res_output]}\n"
                f"Config 2: {desc}"
            )
            raise PreflightError(msg)
        seen_output_dirs[res_output] = desc

        if res_metrics in seen_metrics_files:
            msg = (
                f"Preflight check failed: Duplicate metrics destination detected.\n"
                f"Path: {spec.metrics_file}\n"
                f"Config 1: {seen_metrics_files[res_metrics]}\n"
                f"Config 2: {desc}"
            )
            raise PreflightError(msg)
        seen_metrics_files[res_metrics] = desc

    # 2. Check existence on disk
    existing_destinations: list[str] = []
    for spec in specs:
        desc = (
            f"{spec.category}/{spec.backbone}/ratio_"
            f"{format_ratio(spec.coreset_sampling_ratio)}"
        )
        if spec.output_dir.exists():
            existing_destinations.append(
                f"[Artifact Dir Exists] {spec.output_dir} (from config: {desc})"
            )
        if spec.metrics_file.exists():
            existing_destinations.append(
                f"[Metrics File Exists] {spec.metrics_file} (from config: {desc})"
            )

    if existing_destinations:
        msg = (
            "Preflight check failed: Intended destination(s) already exist on disk.\n"
            "Execution aborted before running any models to prevent overwriting or "
            "silently reusing outputs.\n"
            "Conflicting destinations:\n  "
            + "\n  ".join(existing_destinations)
            + "\nTo proceed, clean up or archive existing outputs."
        )
        raise PreflightError(msg)


def run_experiments(
    specs: Sequence[ExperimentSpec],
    dry_run: bool = False,
) -> list[ExperimentResult]:
    """Execute or dry-run a batch of experiment specifications.

    Args:
        specs: Experiment configurations to run.
        dry_run: When True, list planned runs without invoking model training.

    Returns:
        List of ExperimentResult instances recording outcome per configuration.

    Raises:
        PreflightError: If preflight validation fails when dry_run is False.
    """
    if dry_run:
        print(
            f"=== Experiment Runner Dry-Run "
            f"({len(specs)} configuration{'s' if len(specs) != 1 else ''}) ==="
        )
        for i, spec in enumerate(specs, start=1):
            norm_r = format_ratio(spec.coreset_sampling_ratio)
            print(
                f"[{i}/{len(specs)}] category: {spec.category}, "
                f"backbone: {spec.backbone}, ratio: {norm_r}\n"
                f"     artifact_dir: {spec.output_dir}\n"
                f"     metrics_file: {spec.metrics_file}"
            )
        print("\nDry-run completed. No models were trained or evaluated.")
        return []

    run_preflight_checks(specs)

    results: list[ExperimentResult] = []
    for i, spec in enumerate(specs, start=1):
        norm_r = format_ratio(spec.coreset_sampling_ratio)
        logger.info(
            "Starting experiment [%d/%d]: category=%s, backbone=%s, ratio=%s",
            i,
            len(specs),
            spec.category,
            spec.backbone,
            norm_r,
        )
        try:
            metrics = train_model(
                config=spec.config,
                output_dir=spec.output_dir,
                metrics_file=spec.metrics_file,
            )
            results.append(
                ExperimentResult(spec=spec, status="success", metrics=metrics)
            )
        except Exception as err:
            logger.error(
                "Experiment [%d/%d] failed for category=%s: %s",
                i,
                len(specs),
                spec.category,
                err,
                exc_info=True,
            )
            results.append(ExperimentResult(spec=spec, status="failed", error=str(err)))
            break

    return results


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for experiment runner."""
    parser = argparse.ArgumentParser(
        description=(
            "Run MVTec AD Patchcore experiment matrix with preflight isolation."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "List planned experiment configurations and destinations without training."
        ),
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run only initial pilot configuration (bottle + wide_resnet50_2 + 0.01).",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=Path("params.yaml"),
        help="Path to params.yaml configuration file (default: params.yaml).",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter matrix to a specific category (bottle, screw, capsule).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Filter matrix to a specific backbone (wide_resnet50_2, resnet50).",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="Filter matrix to a specific coreset sampling ratio (0.01, 0.1).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root path to MVTec AD dataset directory. Overrides params.yaml.",
    )
    return parser.parse_args(args)


def validate_cli_selections(
    category: str | None = None,
    backbone: str | None = None,
    ratio: float | None = None,
) -> None:
    """Validate CLI filter selections against supported matrix domains.

    Args:
        category: Optional category filter.
        backbone: Optional backbone filter.
        ratio: Optional coreset ratio filter.

    Raises:
        ValueError: If any provided selection is outside supported values.
    """
    if category is not None and category not in CATEGORIES:
        msg = (
            f"Invalid category '{category}'. "
            f"Supported categories are: {', '.join(CATEGORIES)}."
        )
        raise ValueError(msg)
    if backbone is not None and backbone not in BACKBONES:
        msg = (
            f"Invalid backbone '{backbone}'. "
            f"Supported backbones are: {', '.join(BACKBONES)}."
        )
        raise ValueError(msg)
    if ratio is not None and not any(
        math.isclose(ratio, r, abs_tol=1e-5) for r in CORESET_RATIOS
    ):
        msg = (
            f"Invalid ratio '{ratio}'. "
            f"Supported ratios are: {', '.join(str(r) for r in CORESET_RATIOS)}."
        )
        raise ValueError(msg)


def main(argv: list[str] | None = None) -> int:
    """Execute PatchCore experiment runner CLI.

    Args:
        argv: Command-line arguments.

    Returns:
        0 on success, non-zero on error.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)

    try:
        validate_cli_selections(
            category=args.category,
            backbone=args.backbone,
            ratio=args.ratio,
        )

        base_config = load_params(params_path=args.params)
        if args.data_root is not None:
            base_config = dataclasses.replace(base_config, data_root=args.data_root)

        categories = [args.category] if args.category else CATEGORIES
        backbones = [args.backbone] if args.backbone else BACKBONES
        ratios = [args.ratio] if args.ratio is not None else CORESET_RATIOS

        specs = build_experiment_specs(
            base_config=base_config,
            categories=categories,
            backbones=backbones,
            ratios=ratios,
            artifacts_root=DEFAULT_ARTIFACTS_ROOT,
            metrics_root=DEFAULT_METRICS_ROOT,
        )

        if args.pilot:
            specs = select_pilot_spec(specs)

        results = run_experiments(specs=specs, dry_run=args.dry_run)
        if args.dry_run:
            return 0

        failed_runs = [r for r in results if r.status == "failed"]
        if failed_runs:
            logger.error(
                "Experiment batch halted: %d failure(s) encountered.",
                len(failed_runs),
            )
            return 1

        print(f"\nAll {len(results)} experiment(s) completed successfully.")
        return 0
    except (PreflightError, ValueError, FileNotFoundError) as err:
        logger.error("%s", err)
        return 1
    except Exception as err:
        logger.error("Experiment runner failed: %s", err, exc_info=True)
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
