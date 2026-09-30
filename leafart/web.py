"""Export small WebP leaf sprites and a manifest for the browser screensaver.

The source leaves are several megabytes each; the screensaver only ever draws
them at a couple of hundred pixels, so each one is trimmed, downscaled and
saved as lossy WebP with alpha (roughly 10-20 KB). ``leaves.json`` lists every
sprite with its size and average colour, sorted from purple-red to yellow.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from .color import leaf_color
from .images import list_images, trim_transparent


def run(leaves_dir: Path, out_dir: Path, max_px: int = 256, quality: int = 80) -> None:
    paths = list_images(leaves_dir)
    if not paths:
        print(f"No leaves found in {leaves_dir}")
        return

    sprite_dir = out_dir / "leaves"
    sprite_dir.mkdir(parents=True, exist_ok=True)
    for old in sprite_dir.glob("*.webp"):
        old.unlink()

    entries = []
    for path in paths:
        with Image.open(path) as img:
            img = trim_transparent(img.convert("RGBA"))
            color = leaf_color(img)
            img.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
            entries.append((color, img))

    entries.sort(key=lambda e: e[0].wrapped_hue)
    manifest = []
    total_bytes = 0
    for i, (color, img) in enumerate(entries):
        name = f"leaf_{i:03d}.webp"
        img.save(sprite_dir / name, "WEBP", quality=quality, method=6)
        total_bytes += (sprite_dir / name).stat().st_size
        manifest.append({"src": f"leaves/{name}", "w": img.width, "h": img.height,
                         "color": color.hex, "hue": round(color.wrapped_hue, 1)})

    (out_dir / "leaves.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"Wrote {len(manifest)} sprites ({total_bytes / 1e6:.1f} MB) and leaves.json to {out_dir}")
