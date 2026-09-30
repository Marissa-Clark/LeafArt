"""Stage 2: flag crops that are not one clean leaf.

Leaves that overlapped on the scanner bed come out of extraction as a single
blob, and leaves that ran off the edge of the page come out clipped. This
stage moves:

* likely multi-leaf crops into ``multiple_dir`` (to be split in stage 3), and
* crops with a straight, cut-off edge into ``discard_dir``.

A crop counts as "multiple" if any of these fire:
  - more than one sizable connected component,
  - two or more distance-transform peaks (two blade centres in one blob),
  - several deep convexity defects (the notches where two leaves meet).
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .images import alpha_mask, list_images, load_rgba


@dataclass
class TriageSettings:
    min_component_area_px: int = 500
    peak_dist_threshold: float = 0.20   # lower = more aggressive multi-leaf detection
    min_peak_area_px: int = 3
    defect_depth_frac: float = 0.03     # notch depth as a fraction of the image diagonal
    defect_count_threshold: int = 2
    cutoff_border_frac: float = 0.30    # share of an image border covered by leaf


def count_components(mask: np.ndarray, min_area: int) -> int:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return int(sum(stats[i, cv2.CC_STAT_AREA] >= min_area for i in range(1, n)))


def count_distance_peaks(mask: np.ndarray, s: TriageSettings) -> int:
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    high = dist_norm > s.peak_dist_threshold
    kernel = np.ones((3, 3), np.uint8)
    local_max = dist == cv2.dilate(dist, kernel)
    peaks = (high & local_max).astype(np.uint8) * 255
    peaks = cv2.morphologyEx(peaks, cv2.MORPH_OPEN, kernel)
    return count_components(peaks, s.min_peak_area_px)


def count_deep_defects(mask: np.ndarray, s: TriageSettings) -> int:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 4:
        return 0
    hull = cv2.convexHull(cnt, returnPoints=False)
    if hull is None or len(hull) < 4:
        return 0
    try:
        defects = cv2.convexityDefects(cnt, hull)
    except cv2.error:  # self-intersecting contours occasionally upset OpenCV
        return 0
    if defects is None:
        return 0
    h, w = mask.shape
    min_depth = s.defect_depth_frac * np.hypot(h, w)
    depths = defects[:, 0, 3] / 256.0  # OpenCV stores depth as fixed-point * 256
    return int((depths >= min_depth).sum())


def is_cut_off(mask: np.ndarray, s: TriageSettings) -> bool:
    m = mask > 0
    borders = (m[0, :], m[-1, :], m[:, 0], m[:, -1])
    return max(b.mean() for b in borders) >= s.cutoff_border_frac


def classify(mask: np.ndarray, s: TriageSettings) -> str:
    """Return ``"multiple"``, ``"cutoff"`` or ``"ok"``."""
    if (
        count_components(mask, s.min_component_area_px) > 1
        or count_distance_peaks(mask, s) >= 2
        or count_deep_defects(mask, s) >= s.defect_count_threshold
    ):
        return "multiple"
    if is_cut_off(mask, s):
        return "cutoff"
    return "ok"


def run(leaves_dir: Path, multiple_dir: Path, discard_dir: Path,
        s: TriageSettings | None = None, dry_run: bool = False) -> None:
    s = s or TriageSettings()
    paths = list_images(leaves_dir)
    counts = {"ok": 0, "multiple": 0, "cutoff": 0}
    kernel = np.ones((3, 3), np.uint8)

    for i, path in enumerate(paths, 1):
        mask = cv2.morphologyEx(alpha_mask(load_rgba(path), threshold=0), cv2.MORPH_OPEN, kernel)
        verdict = classify(mask, s)
        counts[verdict] += 1
        if verdict != "ok":
            dest = multiple_dir if verdict == "multiple" else discard_dir
            print(f"[{verdict}] {path.name} -> {dest}")
            if not dry_run:
                dest.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), dest / path.name)
        if i % 50 == 0:
            print(f"  checked {i}/{len(paths)}")

    action = "would move" if dry_run else "moved"
    print(f"Checked {len(paths)} crops: {counts['ok']} ok, "
          f"{counts['multiple']} multi-leaf and {counts['cutoff']} cut-off ({action}).")
