"""Radial colour mosaic: purple-red leaves on the rim, yellow leaves at the centre.

Each leaf is ranked by hue. Rank sets its target distance from the centre, so
the leaves form a radial gradient (the colour target is a disc fading from
yellow in the middle to deep purple at the edge). Each leaf is placed several
times at random angles near its target radius; rejection sampling keeps
copies from stacking. Every copy is turned so its tip points away from the
centre, which gives the piece its bursting-outward look. Finally, extra copies
of the outermost leaves fill the corners so the square frame is covered.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .color import leaf_color
from .images import list_images, trim_transparent


@dataclass
class MosaicSettings:
    size_px: int = 3000
    leaf_frac: float = 0.06          # leaf size relative to the canvas
    copies_per_leaf: int = 3
    corner_copies: int = 6           # extra copies of each rim leaf for the corners
    r_min: float = 0.05              # radius (in half-canvas units) for the most yellow leaf
    r_max: float = 0.95              # radius for the most purple leaf
    r_jitter: float = 0.06
    min_dist: float = 0.08           # minimum spacing between copies
    max_tries: int = 60
    angle_jitter_deg: float = 10
    background: str = "black"
    seed: int | None = 7


def _load_sprites(paths: list[Path], leaf_px: int) -> list[tuple[float, Image.Image]]:
    sprites = []
    for i, path in enumerate(paths, 1):
        with Image.open(path) as img:
            img = trim_transparent(img.convert("RGBA"))
            hue = leaf_color(img).wrapped_hue
            img.thumbnail((leaf_px, leaf_px), Image.Resampling.LANCZOS)
            sprites.append((hue, img))
        if i % 50 == 0:
            print(f"  loaded {i}/{len(paths)}")
    return sprites


def render(leaf_paths: list[Path], s: MosaicSettings) -> Image.Image:
    rng = random.Random(s.seed)
    leaf_px = int(s.size_px * s.leaf_frac)
    sprites = sorted(_load_sprites(leaf_paths, leaf_px), key=lambda t: t[0])
    n = len(sprites)

    # Lowest hue (purple-red) -> outer rim, highest hue (yellow) -> centre.
    leaves = [(s.r_max - (i / max(n - 1, 1)) * (s.r_max - s.r_min), sprite)
              for i, (_, sprite) in enumerate(sprites)]

    plan = [leaf for leaf in leaves for _ in range(s.copies_per_leaf)]
    rng.shuffle(plan)

    placements: list[tuple[float, float, Image.Image]] = []
    placed = np.empty((0, 2))
    for base_r, sprite in plan:
        for _ in range(s.max_tries):
            r = min(max(base_r + rng.uniform(-s.r_jitter, s.r_jitter), 0.0), 1.0)
            theta = rng.uniform(0, 2 * math.pi)
            x, y = r * math.cos(theta), r * math.sin(theta)
            if len(placed) == 0 or np.min(np.hypot(placed[:, 0] - x, placed[:, 1] - y)) >= s.min_dist:
                break
        # If no spaced spot was found, keep the last attempt anyway.
        placed = np.vstack([placed, (x, y)])
        placements.append((x, y, sprite))

    rim = [sprite for base_r, sprite in leaves if base_r > s.r_max - 0.12]
    for sprite in rim:
        for _ in range(s.corner_copies):
            while True:
                x, y = rng.uniform(-1.1, 1.1), rng.uniform(-1.1, 1.1)
                if math.hypot(x, y) >= 0.9:
                    break
            placements.append((x, y, sprite))

    canvas = Image.new("RGBA", (s.size_px, s.size_px), s.background)
    half = s.size_px / 2
    for x, y, sprite in placements:
        # Sprites are tip-up (pointing at 90 degrees); turn them to point outward.
        outward = math.degrees(math.atan2(y, x))
        angle = outward - 90 + rng.uniform(-s.angle_jitter_deg, s.angle_jitter_deg)
        leaf = sprite.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
        px = int(half + x * half - leaf.width / 2)
        py = int(half - y * half - leaf.height / 2)
        # paste() with the leaf as its own mask blends correctly onto the
        # opaque background and, unlike alpha_composite(), allows offsets
        # that hang off the canvas edge.
        canvas.paste(leaf, (px, py), leaf)
    return canvas.convert("RGB")


def run(leaves_dir: Path, out_path: Path, s: MosaicSettings | None = None) -> None:
    s = s or MosaicSettings()
    paths = list_images(leaves_dir)
    if not paths:
        print(f"No leaves found in {leaves_dir}")
        return
    print(f"Rendering mosaic from {len(paths)} leaves ...")
    image = render(paths, s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=92)
    print(f"Saved {out_path}")
