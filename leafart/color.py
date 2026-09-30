"""Leaf colour measurement: average colour of the leaf pixels only."""
from __future__ import annotations

import colorsys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class LeafColor:
    rgb: tuple[int, int, int]
    hue: float         # degrees, 0-360
    saturation: float  # 0-100
    lightness: float   # 0-100

    @property
    def wrapped_hue(self) -> float:
        """Hue in [-180, 180) so reds just below 360 sort next to reds at 0."""
        return (self.hue + 180) % 360 - 180

    @property
    def hex(self) -> str:
        return "#%02x%02x%02x" % self.rgb


def leaf_color(img: Image.Image, sample_size: int = 64) -> LeafColor:
    """Mean colour of the opaque pixels, measured on a small thumbnail."""
    small = img.convert("RGBA")
    small.thumbnail((sample_size, sample_size))
    arr = np.asarray(small)
    opaque = arr[..., 3] > 0
    mean = arr[..., :3][opaque].mean(axis=0) if opaque.any() else np.array([255.0, 255.0, 255.0])
    r, g, b = (float(c) / 255 for c in mean)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return LeafColor(tuple(int(round(c)) for c in mean), h * 360, s * 100, l * 100)


def leaf_color_from_path(path: Path) -> LeafColor:
    with Image.open(path) as img:
        return leaf_color(img)
