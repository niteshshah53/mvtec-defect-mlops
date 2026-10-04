from pathlib import Path

import pytest
from PIL import Image

from mvtec_defect.visualize import (
    VisualizationError,
    main,
    resolve_mask_path,
    select_samples,
)


def _image(path: Path, color: tuple[int, int, int] = (100, 100, 100)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), color).save(path)


def _category(root: Path, category: str = "bottle", good_count: int = 5) -> Path:
    category_dir = root / category
    for index in range(good_count):
        _image(category_dir / "test" / "good" / f"{index:03}.png")
    for defect_type in ("broken_large", "contamination"):
        for index in range(3):
            image_path = category_dir / "test" / defect_type / f"{index:03}.png"
            _image(image_path)
            _image(
                category_dir / "ground_truth" / defect_type / f"{index:03}_mask.png",
                (255, 255, 255),
            )
    return category_dir


def test_selection_is_deterministic_and_distributed(tmp_path: Path) -> None:
    category_dir = _category(tmp_path)

    first = select_samples(category_dir, 5)
    second = select_samples(category_dir, 5)

    assert first == second
    assert [path.name for path in first[0]] == [
        "000.png",
        "001.png",
        "002.png",
        "003.png",
        "004.png",
    ]
    assert [item[0] for item in first[1]] == [
        "broken_large",
        "contamination",
        "broken_large",
        "contamination",
        "broken_large",
    ]


def test_mask_path_resolution(tmp_path: Path) -> None:
    image_path = tmp_path / "test" / "001.png"
    mask_dir = tmp_path / "ground_truth" / "scratch"
    _image(image_path)
    _image(mask_dir / "001_mask.png")

    assert resolve_mask_path(image_path, mask_dir) == mask_dir / "001_mask.png"

    with pytest.raises(VisualizationError, match="Missing ground-truth mask"):
        resolve_mask_path(tmp_path / "test" / "missing.png", mask_dir)


def test_cli_basic_execution(tmp_path: Path) -> None:
    root = tmp_path / "mvtec_ad"
    _category(root)
    output = tmp_path / "reports"

    assert (
        main(
            [
                "--root",
                str(root),
                "--categories",
                "bottle",
                "--samples",
                "2",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    with Image.open(output / "bottle.png") as sheet:
        assert sheet.format == "PNG"
        assert sheet.width > 0 and sheet.height > 0


def test_cli_missing_category_and_data(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "--root",
                str(tmp_path / "missing"),
                "--categories",
                "bottle",
                "--output",
                str(tmp_path / "reports"),
            ]
        )
        == 2
    )
    assert "Dataset root does not exist" in capsys.readouterr().err

    root = tmp_path / "mvtec_ad"
    root.mkdir()
    assert (
        main(
            [
                "--root",
                str(root),
                "--categories",
                "bottle",
                "--output",
                str(tmp_path / "reports"),
            ]
        )
        == 2
    )
    assert "Category directory does not exist" in capsys.readouterr().err


def test_missing_mask_is_reported(tmp_path: Path) -> None:
    category_dir = _category(tmp_path)
    (category_dir / "ground_truth" / "broken_large" / "000_mask.png").unlink()

    with pytest.raises(VisualizationError, match="Missing ground-truth mask"):
        select_samples(category_dir, 5)
