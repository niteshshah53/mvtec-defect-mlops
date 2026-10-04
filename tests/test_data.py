"""Tests for MVTec AD dataset validation and summary."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mvtec_defect.data import (
    DatasetValidationError,
    main,
    summarize_dataset,
)


def _create_fake_category(
    category_dir: Path,
    defects: dict[str, tuple[int, int]] | None = None,
    train_good_count: int = 3,
    test_good_count: int = 2,
) -> None:
    """Create a minimal fake MVTec AD folder structure for testing.

    Args:
        category_dir: Directory where category structure will be created.
        defects: Mapping of defect name to (test_image_count, mask_count).
        train_good_count: Number of fake images in train/good.
        test_good_count: Number of fake images in test/good.
    """
    if defects is None:
        defects = {"broken": (2, 2), "contamination": (1, 1)}

    train_good = category_dir / "train" / "good"
    train_good.mkdir(parents=True, exist_ok=True)
    for i in range(train_good_count):
        (train_good / f"{i:03d}.png").touch()

    test_good = category_dir / "test" / "good"
    test_good.mkdir(parents=True, exist_ok=True)
    for i in range(test_good_count):
        (test_good / f"{i:03d}.png").touch()

    for defect, (img_cnt, mask_cnt) in defects.items():
        defect_dir = category_dir / "test" / defect
        defect_dir.mkdir(parents=True, exist_ok=True)
        for i in range(img_cnt):
            (defect_dir / f"{i:03d}.png").touch()

        gt_dir = category_dir / "ground_truth" / defect
        gt_dir.mkdir(parents=True, exist_ok=True)
        for i in range(mask_cnt):
            (gt_dir / f"{i:03d}_mask.png").touch()


def test_summarize_dataset_valid_single_category(tmp_path: Path) -> None:
    """Validate and count a single fake category correctly."""
    cat_dir = tmp_path / "bottle"
    _create_fake_category(
        cat_dir,
        defects={"broken": (3, 3), "contamination": (2, 2)},
        train_good_count=5,
        test_good_count=4,
    )

    summary = summarize_dataset(root=tmp_path, categories=["bottle"])

    assert isinstance(summary, dict)
    assert "bottle" in summary
    cat = summary["bottle"]

    assert cat["train_good"] == 5
    assert cat["test_good"] == 4
    assert cat["test_defects"] == {"broken": 3, "contamination": 2}
    assert cat["test_defects_total"] == 5
    assert cat["test_total"] == 9
    assert cat["ground_truth_masks"] == 5
    assert cat["ground_truth_defects"] == {"broken": 3, "contamination": 2}


def test_summarize_dataset_multiple_categories(tmp_path: Path) -> None:
    """Validate and summarize multiple fake categories."""
    _create_fake_category(tmp_path / "bottle", train_good_count=4, test_good_count=2)
    _create_fake_category(tmp_path / "screw", train_good_count=6, test_good_count=3)

    summary = summarize_dataset(root=tmp_path, categories=["bottle", "screw"])

    assert set(summary.keys()) == {"bottle", "screw"}
    assert summary["bottle"]["train_good"] == 4
    assert summary["screw"]["train_good"] == 6


def test_root_directory_not_found(tmp_path: Path) -> None:
    """Raise DatasetValidationError when root does not exist."""
    non_existent = tmp_path / "non_existent_dir"
    with pytest.raises(DatasetValidationError, match="root directory does not exist"):
        summarize_dataset(root=non_existent, categories=["bottle"])


def test_root_is_file(tmp_path: Path) -> None:
    """Raise DatasetValidationError when root is a file."""
    file_root = tmp_path / "file.txt"
    file_root.touch()
    with pytest.raises(DatasetValidationError, match="root is not a directory"):
        summarize_dataset(root=file_root, categories=["bottle"])


def test_empty_categories(tmp_path: Path) -> None:
    """Raise DatasetValidationError when categories is empty."""
    with pytest.raises(DatasetValidationError, match="No categories specified"):
        summarize_dataset(root=tmp_path, categories=[])


def test_missing_category_directory(tmp_path: Path) -> None:
    """Raise DatasetValidationError when category folder is missing."""
    with pytest.raises(
        DatasetValidationError, match="Category directory does not exist"
    ):
        summarize_dataset(root=tmp_path, categories=["missing_cat"])


def test_category_is_file(tmp_path: Path) -> None:
    """Raise DatasetValidationError when category path is a file."""
    cat_file = tmp_path / "bottle"
    cat_file.touch()
    with pytest.raises(
        DatasetValidationError, match="Category path is not a directory"
    ):
        summarize_dataset(root=tmp_path, categories=["bottle"])


def test_missing_train_directory(tmp_path: Path) -> None:
    """Raise DatasetValidationError when train directory is missing."""
    cat_dir = tmp_path / "bottle"
    cat_dir.mkdir()
    with pytest.raises(DatasetValidationError, match="missing 'train' directory"):
        summarize_dataset(root=tmp_path, categories=["bottle"])


def test_missing_train_good_directory(tmp_path: Path) -> None:
    """Raise DatasetValidationError when train/good directory is missing."""
    cat_dir = tmp_path / "bottle"
    (cat_dir / "train").mkdir(parents=True)
    with pytest.raises(DatasetValidationError, match="missing 'train/good' directory"):
        summarize_dataset(root=tmp_path, categories=["bottle"])


def test_empty_train_good_directory(tmp_path: Path) -> None:
    """Raise DatasetValidationError when train/good contains no images."""
    cat_dir = tmp_path / "bottle"
    (cat_dir / "train" / "good").mkdir(parents=True)
    with pytest.raises(DatasetValidationError, match="contains no images in"):
        summarize_dataset(root=tmp_path, categories=["bottle"])


def test_missing_test_directory(tmp_path: Path) -> None:
    """Raise DatasetValidationError when test directory is missing."""
    cat_dir = tmp_path / "bottle"
    train_good = cat_dir / "train" / "good"
    train_good.mkdir(parents=True)
    (train_good / "000.png").touch()

    with pytest.raises(DatasetValidationError, match="missing 'test' directory"):
        summarize_dataset(root=tmp_path, categories=["bottle"])


def test_missing_test_good_directory(tmp_path: Path) -> None:
    """Raise DatasetValidationError when test/good directory is missing."""
    cat_dir = tmp_path / "bottle"
    train_good = cat_dir / "train" / "good"
    train_good.mkdir(parents=True)
    (train_good / "000.png").touch()
    (cat_dir / "test").mkdir()

    with pytest.raises(DatasetValidationError, match="missing 'test/good' directory"):
        summarize_dataset(root=tmp_path, categories=["bottle"])


def test_empty_test_good_directory(tmp_path: Path) -> None:
    """Raise DatasetValidationError when test/good contains no images."""
    cat_dir = tmp_path / "bottle"
    train_good = cat_dir / "train" / "good"
    train_good.mkdir(parents=True)
    (train_good / "000.png").touch()
    (cat_dir / "test" / "good").mkdir(parents=True)

    with pytest.raises(DatasetValidationError, match="contains no images in"):
        summarize_dataset(root=tmp_path, categories=["bottle"])


def test_missing_ground_truth_directory(tmp_path: Path) -> None:
    """Raise DatasetValidationError when ground_truth directory is missing."""
    cat_dir = tmp_path / "bottle"
    train_good = cat_dir / "train" / "good"
    train_good.mkdir(parents=True)
    (train_good / "000.png").touch()
    test_good = cat_dir / "test" / "good"
    test_good.mkdir(parents=True)
    (test_good / "000.png").touch()

    with pytest.raises(
        DatasetValidationError, match="missing 'ground_truth' directory"
    ):
        summarize_dataset(root=tmp_path, categories=["bottle"])


def test_no_defect_directories_in_test(tmp_path: Path) -> None:
    """Raise DatasetValidationError when test contains no defect directories."""
    cat_dir = tmp_path / "bottle"
    train_good = cat_dir / "train" / "good"
    train_good.mkdir(parents=True)
    (train_good / "000.png").touch()
    test_good = cat_dir / "test" / "good"
    test_good.mkdir(parents=True)
    (test_good / "000.png").touch()
    (cat_dir / "ground_truth").mkdir(parents=True)

    with pytest.raises(DatasetValidationError, match="has no defect directories under"):
        summarize_dataset(root=tmp_path, categories=["bottle"])


def test_ground_truth_unexpected_good_dir(tmp_path: Path) -> None:
    """Raise DatasetValidationError when ground_truth contains 'good'."""
    cat_dir = tmp_path / "bottle"
    _create_fake_category(cat_dir)
    (cat_dir / "ground_truth" / "good").mkdir(parents=True)

    with pytest.raises(
        DatasetValidationError, match="unexpected 'good' directory under ground_truth"
    ):
        summarize_dataset(root=tmp_path, categories=["bottle"])


def test_ground_truth_missing_defect_folder(tmp_path: Path) -> None:
    """Raise DatasetValidationError when ground_truth is missing a defect folder."""
    cat_dir = tmp_path / "bottle"
    _create_fake_category(cat_dir, defects={"broken": (1, 1), "contamination": (1, 1)})
    shutil.rmtree(cat_dir / "ground_truth" / "contamination")

    with pytest.raises(
        DatasetValidationError, match="missing ground_truth directories"
    ):
        summarize_dataset(root=tmp_path, categories=["bottle"])


def test_ground_truth_extra_defect_folder(tmp_path: Path) -> None:
    """Raise DatasetValidationError when ground_truth has unreferenced defect."""
    cat_dir = tmp_path / "bottle"
    _create_fake_category(cat_dir, defects={"broken": (1, 1)})
    extra_gt = cat_dir / "ground_truth" / "scratch"
    extra_gt.mkdir(parents=True)
    (extra_gt / "000_mask.png").touch()

    with pytest.raises(
        DatasetValidationError, match="ground_truth directories without corresponding"
    ):
        summarize_dataset(root=tmp_path, categories=["bottle"])


def test_empty_defect_folder_in_test(tmp_path: Path) -> None:
    """Raise DatasetValidationError when a defect folder in test is empty."""
    cat_dir = tmp_path / "bottle"
    _create_fake_category(cat_dir, defects={"broken": (1, 1)})
    (cat_dir / "test" / "broken" / "000.png").unlink()

    with pytest.raises(DatasetValidationError, match="has no images in"):
        summarize_dataset(root=tmp_path, categories=["bottle"])


def test_empty_defect_folder_in_ground_truth(tmp_path: Path) -> None:
    """Raise DatasetValidationError when a defect folder in ground_truth is empty."""
    cat_dir = tmp_path / "bottle"
    _create_fake_category(cat_dir, defects={"broken": (1, 1)})
    (cat_dir / "ground_truth" / "broken" / "000_mask.png").unlink()

    with pytest.raises(DatasetValidationError, match="has no masks in"):
        summarize_dataset(root=tmp_path, categories=["bottle"])


def test_defect_image_mask_count_mismatch(tmp_path: Path) -> None:
    """Raise DatasetValidationError when defect image count != mask count."""
    cat_dir = tmp_path / "bottle"
    _create_fake_category(cat_dir, defects={"broken": (2, 1)})

    with pytest.raises(DatasetValidationError, match="image and mask counts differ"):
        summarize_dataset(root=tmp_path, categories=["bottle"])


def test_cli_main_success(tmp_path: Path) -> None:
    """Verify CLI main writes summary JSON to specified output path."""
    fake_root = tmp_path / "mvtec"
    _create_fake_category(fake_root / "bottle", train_good_count=3, test_good_count=2)
    output_path = tmp_path / "custom_reports" / "summary.json"

    ret = main(
        [
            "--root",
            str(fake_root),
            "--categories",
            "bottle",
            "--output",
            str(output_path),
        ]
    )

    assert ret == 0
    assert output_path.is_file()

    with open(output_path, encoding="utf-8") as f:
        data = json.load(f)

    assert "bottle" in data
    assert data["bottle"]["train_good"] == 3
    assert data["bottle"]["test_good"] == 2


def test_cli_main_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI main prints error and returns 1 when validation fails."""
    ret = main(["--root", str(tmp_path / "missing"), "--categories", "bottle"])
    assert ret == 1
    captured = capsys.readouterr()
    assert "Dataset validation failed" in captured.err


def test_cli_subprocess_run(tmp_path: Path) -> None:
    """Verify python -m mvtec_defect.data executes as a module from shell."""
    fake_root = tmp_path / "mvtec"
    _create_fake_category(fake_root / "bottle", train_good_count=2, test_good_count=1)
    output_file = tmp_path / "reports" / "data_summary.json"

    cmd = [
        sys.executable,
        "-m",
        "mvtec_defect.data",
        "--root",
        str(fake_root),
        "--categories",
        "bottle",
        "--output",
        str(output_file),
    ]

    result = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Dataset summary successfully written" in result.stdout
    assert output_file.exists()
