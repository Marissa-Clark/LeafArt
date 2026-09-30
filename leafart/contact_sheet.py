"""Quality check: tile every leaf in a folder onto one image, ordered by colour.

Useful after each stage to spot bad crops at a glance without opening
hundreds of files.
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image

from .color import leaf_color
from .images import list_images, trim_transparent


def run(leaves_dir: Path, out_path: Path, cell_px: int = 128, columns: int = 20,
        background: str = "white") -> None:
    paths = list_images(leaves_dir)
    if not paths:
        print(f"No images found in {leaves_dir}")
        return

    thumbs = []
    for path in paths:
        with Image.open(path) as img:
            img = trim_transparent(img.convert("RGBA"))
            hue = leaf_color(img).wrapped_hue
            img.thumbnail((cell_px - 8, cell_px - 8))
            thumbs.append((hue, img))
    thumbs.sort(key=lambda t: t[0])

    columns = min(columns, len(thumbs))
    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGBA", (columns * cell_px, rows * cell_px), background)
    for i, (_, img) in enumerate(thumbs):
        x = (i % columns) * cell_px + (cell_px - img.width) // 2
        y = (i // columns) * cell_px + (cell_px - img.height) // 2
        sheet.alpha_composite(img, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.convert("RGB").save(out_path, quality=90)
    print(f"Saved a {columns}x{rows} contact sheet of {len(thumbs)} leaves to {out_path}")
