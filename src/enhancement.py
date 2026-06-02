"""Image enhancement: CLAHE, global histogram equalization, adaptive gamma."""

from __future__ import annotations

import cv2
import numpy as np

from src.preprocessing import to_uint8

ENHANCEMENT_METHODS = ("none", "clahe", "histeq", "gamma")


def clahe_enhance(img_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """CLAHE on L channel in LAB color space."""
    img_u8 = to_uint8(img_bgr)
    lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def histeq_global(img_bgr: np.ndarray) -> np.ndarray:
    """Global histogram equalization on V channel in HSV."""
    img_u8 = to_uint8(img_bgr)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def gamma_adaptive(img_bgr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Adaptive gamma correction from mean L in LAB.
    gamma = log(128) / log(mean_L + eps)
    """
    img_u8 = to_uint8(img_bgr)
    lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB)
    mean_l = float(np.mean(lab[:, :, 0])) + eps
    gamma = np.log(128.0) / np.log(mean_l)

    inv_gamma = 1.0 / gamma
    # Clip the table BEFORE casting to uint8: when gamma < 0 (mean_L < 1),
    # inv_gamma is negative and the raw float values overflow uint8 range,
    # wrapping silently to wrong pixel values instead of saturating at 255.
    table_f = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)])
    table = np.clip(table_f, 0, 255).astype(np.uint8)
    return cv2.LUT(img_u8, table)


def apply_enhancement(img_bgr: np.ndarray, method: str = "none") -> np.ndarray:
    """Apply enhancement method. Returns uint8 BGR."""
    method = method.lower().strip()
    if method in ("none", ""):
        return to_uint8(img_bgr)
    if method == "clahe":
        return clahe_enhance(img_bgr)
    if method == "histeq":
        return histeq_global(img_bgr)
    if method == "gamma":
        return gamma_adaptive(img_bgr)

    raise ValueError(f"Unknown enhancement method: {method}. Use one of {ENHANCEMENT_METHODS}")
