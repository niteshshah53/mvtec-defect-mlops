"""Dataset inspection, validation, and summary utilities for MVTec AD."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


class DatasetValidationError(ValueError, FileNotFoundError):
    """Raised when the MVTec AD dataset directory structure is missing or invalid."""


def _count_images(directory: Path) -> int:
    """Count image files with valid extensions in the specified directory."""
    if not directory.is_dir():
        return 0
    return sum(
        1
        for item in directory.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def summarize_dataset(
    root: str | Path,
    categories: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Validate MVTec AD folder structure and count images and masks.

    Args:
        root: Root path to the MVTec AD dataset directory.
        categories: Sequence of category names to inspect and summarize.

    Returns:
        Dictionary mapping each category to its summary statistics.

    Raises:
        DatasetValidationError: If dataset root, category, or subdirectories
            are missing or violate the MVTec AD layout, if required folders
            contain no image files, or if defect image and mask counts mismatch.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise DatasetValidationError(
            f"Dataset root directory does not exist: {root_path}"
        )
    if not root_path.is_dir():
        raise DatasetValidationError(f"Dataset root is not a directory: {root_path}")

    if not categories:
        raise DatasetValidationError("No categories specified to summarize.")

    summary: dict[str, dict[str, Any]] = {}

    for category in categories:
        cat_dir = root_path / category
        if not cat_dir.exists():
            raise DatasetValidationError(
                f"Category directory does not exist for '{category}': {cat_dir}"
            )
        if not cat_dir.is_dir():
            raise DatasetValidationError(
                f"Category path is not a directory for '{category}': {cat_dir}"
            )

        # Validate train folder
        train_dir = cat_dir / "train"
        if not train_dir.is_dir():
            raise DatasetValidationError(
                f"Category '{category}' is missing 'train' directory: {train_dir}"
            )

        train_good_dir = train_dir / "good"
        if not train_good_dir.is_dir():
            raise DatasetValidationError(
                f"Category '{category}' is missing 'train/good' directory: "
                f"{train_good_dir}"
            )

        train_good_count = _count_images(train_good_dir)
        if train_good_count == 0:
            raise DatasetValidationError(
                f"Category '{category}' contains no images in {train_good_dir}"
            )

        # Validate test folder
        test_dir = cat_dir / "test"
        if not test_dir.is_dir():
            raise DatasetValidationError(
                f"Category '{category}' is missing 'test' directory: {test_dir}"
            )

        test_good_dir = test_dir / "good"
        if not test_good_dir.is_dir():
            raise DatasetValidationError(
                f"Category '{category}' is missing 'test/good' directory: "
                f"{test_good_dir}"
            )

        test_good_count = _count_images(test_good_dir)
        if test_good_count == 0:
            raise DatasetValidationError(
                f"Category '{category}' contains no images in {test_good_dir}"
            )

        # Validate ground_truth folder
        gt_dir = cat_dir / "ground_truth"
        if not gt_dir.is_dir():
            raise DatasetValidationError(
                f"Category '{category}' is missing 'ground_truth' directory: {gt_dir}"
            )

        # Identify defect types from test subdirectories (excluding 'good')
        test_subdirs = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
        defect_types = [d for d in test_subdirs if d != "good"]
        if not defect_types:
            raise DatasetValidationError(
                f"Category '{category}' has no defect directories under {test_dir}"
            )

        gt_subdirs = sorted([d.name for d in gt_dir.iterdir() if d.is_dir()])
        if "good" in gt_subdirs:
            raise DatasetValidationError(
                f"Category '{category}' has unexpected 'good' directory under "
                f"ground_truth: {gt_dir / 'good'}"
            )

        missing_gt = sorted(set(defect_types) - set(gt_subdirs))
        if missing_gt:
            raise DatasetValidationError(
                f"Category '{category}' is missing ground_truth directories for "
                f"defect types: {missing_gt}"
            )

        extra_gt = sorted(set(gt_subdirs) - set(defect_types))
        if extra_gt:
            raise DatasetValidationError(
                f"Category '{category}' has ground_truth directories without "
                f"corresponding test defect: {extra_gt}"
            )

        # Count images in test defects and masks in ground_truth
        test_defects: dict[str, int] = {}
        gt_masks: dict[str, int] = {}

        for defect in defect_types:
            test_defect_dir = test_dir / defect
            defect_img_count = _count_images(test_defect_dir)
            if defect_img_count == 0:
                raise DatasetValidationError(
                    f"Category '{category}' defect '{defect}' has no images in "
                    f"{test_defect_dir}"
                )

            gt_defect_dir = gt_dir / defect
            defect_mask_count = _count_images(gt_defect_dir)
            if defect_mask_count == 0:
                raise DatasetValidationError(
                    f"Category '{category}' defect '{defect}' has no masks in "
                    f"{gt_defect_dir}"
                )

            if defect_img_count != defect_mask_count:
                raise DatasetValidationError(
                    f"Category '{category}' defect '{defect}' image and mask counts "
                    f"differ: {defect_img_count} test image(s) vs "
                    f"{defect_mask_count} ground-truth mask(s)"
                )

            test_defects[defect] = defect_img_count
            gt_masks[defect] = defect_mask_count

        total_test_defects = sum(test_defects.values())
        total_gt_masks = sum(gt_masks.values())

        summary[category] = {
            "train_good": train_good_count,
            "test_good": test_good_count,
            "test_defects": test_defects,
            "test_defects_total": total_test_defects,
            "test_total": test_good_count + total_test_defects,
            "ground_truth_masks": total_gt_masks,
            "ground_truth_defects": gt_masks,
        }

    return summary


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for dataset validation and summary."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate MVTec AD folder structure and summarize image/mask counts."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/mvtec_ad"),
        help="Root directory of MVTec AD dataset (default: data/mvtec_ad).",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["bottle", "screw", "capsule"],
        help="Categories to validate and summarize (default: bottle screw capsule).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/data_summary.json"),
        help="Output path for summary JSON (default: reports/data_summary.json).",
    )
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> int:
    """CLI entrypoint for dataset validation and summary."""
    parsed = parse_args(args)
    try:
        summary = summarize_dataset(root=parsed.root, categories=parsed.categories)
        parsed.output.parent.mkdir(parents=True, exist_ok=True)
        with open(parsed.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Dataset summary successfully written to {parsed.output}")
        return 0
    except (DatasetValidationError, ValueError, FileNotFoundError) as exc:
        print(f"Dataset validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
