"""Stage 3: separate touching leaves that triage flagged as "multiple".

Two strategies, tried in order:

1. **Watershed.** Leaf centres are the peaks of the distance transform; they
   seed a marker-based watershed that floods outward until neighbouring
   leaves meet.
2. **Concavity cut.** If watershed finds only one region, look for the concave
   points where two outlines meet, cluster them into two groups, and cut the
   blob along the line between them.

If neither produces at least two leaves, the crop was probably a single leaf
with an unusual shape, so it is moved back to the leaves folder unchanged.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .images import alpha_mask, crop_to_mask, list_images, load_rgba, save_rgba


@dataclass
class SplitSettings:
    marker_frac: float = 0.45        # distance-transform fraction that seeds watershed
    min_region_frac: float = 0.01    # ignore regions smaller than 1% of the blob
    min_region_px: int = 2000
    cut_width_px: int = 25


def watershed_regions(rgb: np.ndarray, mask: np.ndarray, s: SplitSettings) -> list[np.ndarray]:
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    sure_fg = (dist > s.marker_frac * dist.max()).astype(np.uint8) * 255
    sure_bg = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1          # background becomes 1, leaves 2..n
    markers[unknown == 255] = 0    # let watershed decide these pixels
    labels = cv2.watershed(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), markers)

    min_area = max(s.min_region_px, int(s.min_region_frac * np.count_nonzero(mask)))
    regions = []
    for label in np.unique(labels):
        if label <= 1:  # -1 = watershed boundary, 1 = background
            continue
        region = cv2.bitwise_and((labels == label).astype(np.uint8) * 255, mask)
        if np.count_nonzero(region) >= min_area:
            regions.append(region)
    return regions


def concavity_regions(mask: np.ndarray, s: SplitSettings) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea)
    points = cnt[:, 0, :].astype(np.float32)
    on_hull = np.zeros(len(points), bool)
    on_hull[cv2.convexHull(cnt, returnPoints=False).flatten()] = True
    concave = points[~on_hull]
    if len(concave) < 2:
        return []

    # Split the concave points into two clusters along their main axis. Each
    # cluster sits mostly on one leaf, so cutting perpendicular to the line
    # between the cluster centres, through its midpoint, separates the leaves.
    centred = concave - concave.mean(axis=0)
    _, vecs = np.linalg.eigh(centred.T @ centred)
    proj = centred @ vecs[:, -1]
    a, b = concave[proj <= 0], concave[proj > 0]
    if len(a) == 0 or len(b) == 0:
        return []
    p1, p2 = a.mean(axis=0), b.mean(axis=0)

    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return []
    perp = np.array([-direction[1], direction[0]]) / length
    mid = (p1 + p2) / 2
    reach = 2 * max(mask.shape)  # long enough to cross the whole crop
    start = tuple(int(v) for v in mid - perp * reach)
    end = tuple(int(v) for v in mid + perp * reach)

    cut = np.zeros_like(mask)
    cv2.line(cut, start, end, 255, thickness=s.cut_width_px)
    cut = cv2.dilate(cut, np.ones((35, 35), np.uint8))
    separated = cv2.bitwise_and(mask, cv2.bitwise_not(cut))

    n, labels = cv2.connectedComponents(separated)
    regions = [(labels == i).astype(np.uint8) * 255 for i in range(1, n)]
    regions = [r for r in regions if np.count_nonzero(r) >= s.min_region_px]
    return regions if len(regions) >= 2 else []


def split_leaf(rgba: np.ndarray, s: SplitSettings) -> list[np.ndarray]:
    """Return one RGBA crop per leaf found (empty if it could not be split)."""
    mask = alpha_mask(rgba)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    rgb = np.ascontiguousarray(rgba[..., :3])

    regions = watershed_regions(rgb, mask, s)
    if len(regions) < 2:
        regions = concavity_regions(mask, s)
    if len(regions) < 2:
        return []
    crops = [crop_to_mask(rgb, r) for r in regions]
    return [c for c in crops if c is not None]


def run(multiple_dir: Path, leaves_dir: Path, s: SplitSettings | None = None) -> None:
    s = s or SplitSettings()
    originals_dir = multiple_dir / "originals"
    paths = list_images(multiple_dir)
    split_count = kept = 0

    for path in paths:
        crops = split_leaf(load_rgba(path), s)
        if crops:
            for i, crop in enumerate(crops):
                save_rgba(crop, leaves_dir / f"{path.stem}_split_{i:02d}.png")
            originals_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), originals_dir / path.name)
            split_count += 1
            print(f"{path.name}: split into {len(crops)} leaves")
        else:
            shutil.move(str(path), leaves_dir / path.name)
            kept += 1
            print(f"{path.name}: looks like one leaf, moved back")

    print(f"Split {split_count} crops; returned {kept} to {leaves_dir}. "
          f"Originals of split crops are in {originals_dir}.")
