"""Preprocessing: integrity check, resize, normalize, SSR on L channel."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert any image array to uint8 BGR.

    - uint8 input  → returned as-is (zero-copy).
    - float input  → scaled ×255, clipped to [0, 255], cast to uint8.

    This is the single canonical conversion used across the entire pipeline.
    """
    if img.dtype == np.uint8:
        return img
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def check_integrity(path: str | Path) -> bool:
    """Return True if image can be read without corruption."""
    img = cv2.imread(str(path))
    return img is not None and img.size > 0


def resize_image(img: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize image with bilinear interpolation."""
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def normalize_pixels(img: np.ndarray) -> np.ndarray:
    """Normalize uint8 BGR to float [0, 1]."""
    return to_uint8(img).astype(np.float32) / 255.0


def apply_ssr(img_bgr: np.ndarray, sigma: float = 30.0) -> np.ndarray:
    """
    Single-Scale Retinex on L channel (CIELAB) only.
    log(R) = log(I) - log(G_sigma * I)
    Returns uint8 BGR.
    """
    img_u8 = to_uint8(img_bgr)
    lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32) + 1.0

    blurred = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=sigma, sigmaY=sigma) + 1.0
    retinex = np.log(l_channel) - np.log(blurred)
    retinex = retinex - retinex.min()
    if retinex.max() > 0:
        retinex = retinex / retinex.max()
    l_new = (retinex * 255).astype(np.uint8)
    lab[:, :, 0] = l_new
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_from_array(img_bgr: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize + SSR on in-memory BGR image. Returns uint8 BGR."""
    resized = resize_image(img_bgr, size)
    return apply_ssr(resized)


def preprocess_pipeline(path: str | Path, size: tuple[int, int] = (224, 224)) -> np.ndarray | None:
    """
    Full preprocess: load → resize → SSR.
    Returns uint8 BGR or None if file is corrupt / unreadable.

    Reads the file exactly once (previously check_integrity read it once, then
    cv2.imread read it again — now merged into a single read).
    """
    img = cv2.imread(str(path))
    if img is None or img.size == 0:
        return None
    return preprocess_from_array(img, size)
