"""Fruit segmentation: Otsu, morphology, masking, fallback."""

from __future__ import annotations

import cv2
import numpy as np

from src.config import SEGMENTATION_FALLBACK_RATIO as FALLBACK_RATIO
from src.preprocessing import to_uint8


def _largest_contour_mask(binary: np.ndarray) -> np.ndarray:
    """Keep only the largest contour region."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary)
    if not contours:
        return mask
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    return mask


def segment_fruit(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """
    Segment fruit using Otsu on S (HSV) and grayscale, morphology, largest contour.

    Returns:
        masked_img: BGR with background (0,0,0) or full image on fallback
        binary_mask: uint8 mask 0/255
        object_ratio: fraction of foreground pixels
        used_fallback: True if segmentation failed (< 5% object)
    """
    img_u8 = to_uint8(img_bgr)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)

    _, mask_s = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask_g = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    combined = cv2.bitwise_or(mask_s, mask_g)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

    binary_mask = _largest_contour_mask(closed)
    object_ratio = float(np.count_nonzero(binary_mask)) / binary_mask.size

    if object_ratio < FALLBACK_RATIO:
        full_mask = np.ones(img_u8.shape[:2], dtype=np.uint8) * 255
        return img_u8.copy(), full_mask, object_ratio, True

    masked = cv2.bitwise_and(img_u8, img_u8, mask=binary_mask)
    return masked, binary_mask, object_ratio, False


def object_ratio_from_mask(binary_mask: np.ndarray) -> float:
    """Compute foreground ratio from binary mask."""
    return float(np.count_nonzero(binary_mask)) / binary_mask.size
