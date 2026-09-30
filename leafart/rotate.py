"""Stage 4: turn every leaf upright using hand-measured angles.

``leaf_rotations.csv`` has one row per leaf (``filename,angle``). The angle is
in degrees, clockwise, and turns each leaf so its tip points up and its midrib
is vertical. Having every leaf share one orientation lets the mosaic and the
screensaver control which way each leaf faces.
"""
from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from .images import trim_transparent


def read_angles(csv_path: Path) -> dict[str, float]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {row["filename"]: float(row["angle"]) for row in csv.DictReader(f)}


def run(leaves_dir: Path, csv_path: Path, out_dir: Path) -> None:
    angles = read_angles(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = missing = 0

    for filename, angle in angles.items():
        src = leaves_dir / filename
        if not src.exists():
            missing += 1
            continue
        with Image.open(src) as img:
            rotated = img.convert("RGBA").rotate(
                -angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0)
            )
            trim_transparent(rotated).save(out_dir / f"rotated_{src.stem}.png")
        done += 1
        if done % 25 == 0:
            print(f"  rotated {done}")

    print(f"Rotated {done} leaves into {out_dir}"
          + (f" ({missing} listed in the CSV were not found, e.g. discarded)" if missing else ""))
