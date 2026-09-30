"""Stage 1: cut individual leaves out of full-page scans.

Each scan is a sheet of leaves on a white scanner bed. For every page we:

1. Threshold the page (Otsu, nudged upward so pale yellow leaves still count
   as foreground) and clean the mask with a small close/open.
2. Remove stems by shape: a distance transform keeps only pixels far from the
   background ("leaf cores"), then dilating those cores by the same radius
   grows the blades back while thin stems never return.
3. Take each large contour as one leaf, crop it with its mask as alpha, and
   knock out the bright, unsaturated "halo" that scanners leave along edges.

Default sizes are in pixels and tuned for 1200 DPI scans.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .images import list_images, load_rgb, save_rgba


@dataclass
class ExtractSettings:
    threshold_boost: int = 30      # added to Otsu's threshold to catch pale leaves
    stem_radius_px: int = 50       # anything thinner than ~2x this is treated as stem
    min_leaf_area_px: int = 150_000
    padding_px: int = 10
    halo_band_px: int = 400        # how far in from the edge to look for halo
    halo_max_saturation: int = 90
    halo_min_value: int = 170


def leaf_mask(rgb: np.ndarray, s: ExtractSettings) -> np.ndarray:
    """Binary mask of leaf blades (stems removed) for a full page."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(blur, min(255, otsu + s.threshold_boost), 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Morphological opening with a disk of radius r, done with two distance
    # transforms: much faster than cv2.dilate with a 101x101 kernel on a
    # 135-megapixel page, and equivalent up to edge rounding.
    r = s.stem_radius_px
    core = cv2.distanceTransform(mask, cv2.DIST_L2, 5) > r
    grown = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5) <= r
    return cv2.bitwise_and(grown.astype(np.uint8) * 255, mask)


def remove_halo(rgba: np.ndarray, s: ExtractSettings) -> np.ndarray:
    """Make bright grey pixels near the leaf edge transparent."""
    alpha = rgba[..., 3]
    if not alpha.any():
        return rgba
    dist = cv2.distanceTransform((alpha > 0).astype(np.uint8), cv2.DIST_L2, 5)
    edge_band = (alpha > 0) & (dist <= s.halo_band_px)
    hsv = cv2.cvtColor(np.ascontiguousarray(rgba[..., :3]), cv2.COLOR_RGB2HSV)
    halo = edge_band & (hsv[..., 1] < s.halo_max_saturation) & (hsv[..., 2] > s.halo_min_value)
    out = rgba.copy()
    out[..., 3][halo] = 0
    return out


def extract_page(page: Path, out_dir: Path, s: ExtractSettings) -> int:
    """Extract every leaf on one scanned page. Returns the number saved."""
    rgb = load_rgb(page)
    mask = leaf_mask(rgb, s)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = mask.shape
    pad = s.padding_px
    count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) < s.min_leaf_area_px:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + cw + pad, w), min(y + ch + pad, h)
        alpha = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.drawContours(alpha, [cnt], -1, 255, thickness=-1, offset=(-x0, -y0))
        crop = np.dstack([rgb[y0:y1, x0:x1], alpha])
        save_rgba(remove_halo(crop, s), out_dir / f"{page.stem}_leaf_{count:02d}.png")
        count += 1
    return count


def run(scans_dir: Path, out_dir: Path, s: ExtractSettings | None = None) -> None:
    s = s or ExtractSettings()
    pages = list_images(scans_dir)
    if not pages:
        print(f"No scans found in {scans_dir}")
        return
    total = 0
    for i, page in enumerate(pages, 1):
        n = extract_page(page, out_dir, s)
        total += n
        print(f"[{i}/{len(pages)}] {page.name}: {n} leaves")
    print(f"Extracted {total} leaves from {len(pages)} scans into {out_dir}")
