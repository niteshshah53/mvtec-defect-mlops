"""Create deterministic MVTec AD contact sheets."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
TILE_SIZE = (256, 256)
TEXT_HEIGHT = 42
GAP = 8
HEADER_HEIGHT = 24


class VisualizationError(ValueError):
    """Raised when the dataset cannot be visualized."""


def _images_in(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def resolve_mask_path(image_path: Path, ground_truth_dir: Path) -> Path:
    """Resolve the MVTec ground-truth mask corresponding to an image."""
    candidates = sorted(
        path
        for path in ground_truth_dir.glob(f"{image_path.stem}_mask.*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not candidates:
        raise VisualizationError(
            f"Missing ground-truth mask for {image_path.name} in {ground_truth_dir}"
        )
    return candidates[0]


def select_samples(
    category_dir: Path, samples: int
) -> tuple[list[Path], list[tuple[str, Path, Path]]]:
    """Select good images and defect images in deterministic type round-robin order."""
    test_dir = category_dir / "test"
    good_dir = test_dir / "good"
    if not good_dir.is_dir():
        raise VisualizationError(f"Missing test/good directory: {good_dir}")

    good_images = _images_in(good_dir)
    if len(good_images) < samples:
        raise VisualizationError(
            f"Need {samples} good images in {good_dir}, found {len(good_images)}"
        )

    defect_dirs = sorted(
        path for path in test_dir.iterdir() if path.is_dir() and path.name != "good"
    )
    if not defect_dirs:
        raise VisualizationError(f"No defect directories under {test_dir}")

    defect_images = {
        defect_dir.name: _images_in(defect_dir) for defect_dir in defect_dirs
    }
    selected: list[tuple[str, Path, Path]] = []
    for index in range(max(len(images) for images in defect_images.values())):
        for defect_type in sorted(defect_images):
            images = defect_images[defect_type]
            if index < len(images) and len(selected) < samples:
                image_path = images[index]
                mask_path = resolve_mask_path(
                    image_path, category_dir / "ground_truth" / defect_type
                )
                selected.append((defect_type, image_path, mask_path))
        if len(selected) == samples:
            break

    if len(selected) < samples:
        raise VisualizationError(
            f"Need {samples} defect images under {test_dir}, found {len(selected)}"
        )
    return good_images[:samples], selected


def _thumbnail(image: Image.Image) -> Image.Image:
    return ImageOps.contain(image.convert("RGB"), TILE_SIZE, Image.Resampling.LANCZOS)


def _labelled_tile(image: Image.Image, label: str) -> Image.Image:
    tile = Image.new("RGB", (TILE_SIZE[0], TILE_SIZE[1] + TEXT_HEIGHT), "white")
    thumbnail = _thumbnail(image)
    offset = (
        (TILE_SIZE[0] - thumbnail.width) // 2,
        (TILE_SIZE[1] - thumbnail.height) // 2,
    )
    tile.paste(thumbnail, offset)
    ImageDraw.Draw(tile).text((4, TILE_SIZE[1] + 3), label, fill="black")
    return tile


def _defect_overlay(image: Image.Image, mask: Image.Image) -> Image.Image:
    original = image.convert("RGBA")
    mask_rgb = mask.convert("L").resize(original.size, Image.Resampling.NEAREST)
    highlight = Image.new("RGBA", original.size, (255, 0, 0, 0))
    highlight.putalpha(mask_rgb.point(lambda value: min(value, 150)))
    highlight.paste((255, 0, 0, 150), mask=mask_rgb)
    return Image.alpha_composite(original, highlight).convert("RGB")


def create_contact_sheet(category_dir: Path, output_path: Path, samples: int) -> None:
    """Create one contact sheet for a category."""
    good_images, defects = select_samples(category_dir, samples)
    category = category_dir.name
    rows: list[list[Image.Image]] = []

    for image_path in good_images:
        with Image.open(image_path) as image:
            rows.append(
                [
                    _labelled_tile(image, f"{category} | good | {image_path.name}"),
                    _labelled_tile(Image.new("RGB", TILE_SIZE, "white"), "good sample"),
                    _labelled_tile(Image.new("RGB", TILE_SIZE, "white"), "good sample"),
                ]
            )

    for defect_type, image_path, mask_path in defects:
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            rows.append(
                [
                    _labelled_tile(
                        image, f"{category} | {defect_type} | {image_path.name}"
                    ),
                    _labelled_tile(mask, f"{defect_type} | ground truth mask"),
                    _labelled_tile(
                        _defect_overlay(image, mask), f"{defect_type} | overlay"
                    ),
                ]
            )

    tile_width = TILE_SIZE[0]
    tile_height = TILE_SIZE[1] + TEXT_HEIGHT
    sheet = Image.new(
        "RGB",
        (
            3 * tile_width + 4 * GAP,
            HEADER_HEIGHT + len(rows) * tile_height + (len(rows) + 1) * GAP,
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((GAP, 2), f"{category} dataset inspection", fill="black")
    for row_index, row in enumerate(rows):
        y = HEADER_HEIGHT + GAP + row_index * tile_height
        for column_index, tile in enumerate(row):
            x = GAP + column_index * (tile_width + GAP)
            sheet.paste(tile, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG")


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(args)
    if parsed.samples < 1:
        parser.error("--samples must be at least 1")
    return parsed


def main(args: Sequence[str] | None = None) -> int:
    parsed = parse_args(args)
    if not parsed.root.is_dir():
        print(
            f"Dataset root does not exist or is not a directory: {parsed.root}",
            file=sys.stderr,
        )
        return 2
    try:
        for category in parsed.categories:
            category_dir = parsed.root / category
            if not category_dir.is_dir():
                raise VisualizationError(
                    f"Category directory does not exist: {category_dir}"
                )
            create_contact_sheet(
                category_dir, parsed.output / f"{category}.png", parsed.samples
            )
    except VisualizationError as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
