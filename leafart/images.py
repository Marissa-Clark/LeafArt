"""Small image helpers shared by every pipeline stage.

All arrays are RGB(A) channel order, uint8. OpenCV is used for reading because
it is much faster than Pillow on the very large 1200 DPI scans; Pillow is used
for writing because it handles RGBA PNG/WebP cleanly.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# Full-page scans are ~135 megapixels, which trips Pillow's decompression-bomb
# guard. These are our own files, so the guard is not useful here.
Image.MAX_IMAGE_PIXELS = None


def list_images(folder: Path) -> list[Path]:
    """Sorted image files directly inside ``folder`` (non-recursive)."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_rgba(path: Path) -> np.ndarray:
    """Load any image as an (H, W, 4) RGBA array; opaque if it has no alpha."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)


def save_rgba(rgba: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Level 3 is ~3x faster than Pillow's default (6) on these multi-megapixel
    # crops for files only ~10% larger.
    Image.fromarray(rgba, "RGBA").save(path, compress_level=3)


def alpha_mask(rgba: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Binary 0/255 mask of the leaf pixels in an RGBA crop."""
    return (rgba[..., 3] > threshold).astype(np.uint8) * 255


def crop_to_mask(rgb: np.ndarray, mask: np.ndarray, padding: int = 10) -> np.ndarray | None:
    """Crop ``rgb`` to the bounding box of ``mask`` and use the mask as alpha.

    Returns an RGBA array, or ``None`` if the mask is empty.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    h, w = mask.shape[:2]
    y0, y1 = max(ys.min() - padding, 0), min(ys.max() + padding + 1, h)
    x0, x1 = max(xs.min() - padding, 0), min(xs.max() + padding + 1, w)
    return np.dstack([rgb[y0:y1, x0:x1, :3], mask[y0:y1, x0:x1]])


def trim_transparent(img: Image.Image) -> Image.Image:
    """Crop a Pillow RGBA image to its non-transparent pixels."""
    bbox = img.getchannel("A").getbbox()
    return img.crop(bbox) if bbox else img
