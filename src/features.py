"""Manual feature extraction: color, texture, shape."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import regionprops

from src.config import SEGMENTATION_FALLBACK_RATIO as FALLBACK_RATIO
from src.preprocessing import to_uint8

FeatureGroups = Literal["color", "texture", "shape", "all"]
HIST_BINS = 64


def _foreground_mask(img_bgr: np.ndarray) -> np.ndarray | None:
    """Return non-black binary mask, or None when foreground < FALLBACK_RATIO."""
    mask = np.any(img_bgr != [0, 0, 0], axis=-1)
    if mask.sum() < img_bgr.shape[0] * img_bgr.shape[1] * FALLBACK_RATIO:
        return None
    return mask


def get_object_pixels(img_bgr: np.ndarray, segmented: bool = True) -> np.ndarray:
    """
    Return Nx3 BGR pixels for feature extraction.
    Uses mask (non-black) when segmented; fallback to full image if object < 5%.
    """
    if not segmented:
        return img_bgr.reshape(-1, 3)
    mask = _foreground_mask(img_bgr)
    return img_bgr[mask] if mask is not None else img_bgr.reshape(-1, 3)


def _pixels_to_hsv(pixels_bgr: np.ndarray) -> np.ndarray:
    """Convert flat Nx3 BGR pixel array to HSV using a single-row image trick."""
    n = len(pixels_bgr)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    row = pixels_bgr.reshape(1, n, 3).astype(np.uint8)
    hsv_row = cv2.cvtColor(row, cv2.COLOR_BGR2HSV)
    return hsv_row.reshape(n, 3).astype(np.float32)


def extract_color_histogram(img_bgr: np.ndarray, segmented: bool = True) -> np.ndarray:
    """HSV histogram: 64 bins per channel -> 192 dim."""
    pixels = get_object_pixels(img_bgr, segmented)
    hsv = _pixels_to_hsv(pixels)
    if len(hsv) == 0:
        return np.zeros(HIST_BINS * 3, dtype=np.float32)

    feats = []
    for ch in range(3):
        max_val = 180 if ch == 0 else 256
        hist, _ = np.histogram(hsv[:, ch], bins=HIST_BINS, range=(0, max_val), density=True)
        feats.append(hist.astype(np.float32))
    return np.concatenate(feats)


def extract_color_moments(img_bgr: np.ndarray, segmented: bool = True) -> np.ndarray:
    """Mean, std, skewness per HSV channel -> 9 dim."""
    pixels = get_object_pixels(img_bgr, segmented)
    hsv = _pixels_to_hsv(pixels)
    if len(hsv) == 0:
        return np.zeros(9, dtype=np.float32)

    moments = []
    for ch in range(3):
        channel = hsv[:, ch]
        moments.extend(
            [
                float(np.mean(channel)),
                float(np.std(channel)),
                float(skew(channel)) if len(channel) > 2 else 0.0,
            ]
        )
    return np.array(moments, dtype=np.float32)


def extract_glcm_features(img_bgr: np.ndarray, segmented: bool = True) -> np.ndarray:
    """
    GLCM on grayscale: d=[1,2], angles=[0,45,90,135].
    Average contrast, energy, homogeneity, correlation -> 4 dim.
    """
    img_u8 = to_uint8(img_bgr)

    gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
    if segmented:
        mask = _foreground_mask(img_u8)
        if mask is not None:
            gray = gray.copy()
            gray[~mask] = 0

    if gray.max() == 0:
        return np.zeros(4, dtype=np.float32)

    distances = [1, 2]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(
        gray,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True,
    )

    props = ("contrast", "energy", "homogeneity", "correlation")
    values = [float(graycoprops(glcm, p).mean()) for p in props]
    return np.array(values, dtype=np.float32)


def extract_lbp_features(img_bgr: np.ndarray, segmented: bool = True) -> np.ndarray:
    """LBP uniform P=8, R=1 -> 10 dim histogram."""
    img_u8 = to_uint8(img_bgr)

    gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
    if segmented:
        mask = _foreground_mask(img_u8)
        if mask is not None:
            gray = gray.copy()
            gray[~mask] = 0

    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    return hist.astype(np.float32)


def extract_shape_features(binary_mask: np.ndarray) -> np.ndarray | None:
    """
    Shape features from binary mask: area, perimeter (normalized), eccentricity, extent, solidity.
    Returns None if mask is empty.
    """
    labeled = binary_mask.astype(np.uint8)
    if labeled.max() == 0:
        return None

    props_list = regionprops(labeled)
    if not props_list:
        return None

    props = max(props_list, key=lambda p: p.area)
    h, w = binary_mask.shape
    return np.array(
        [
            props.area / (h * w),
            props.perimeter / (h + w),
            props.eccentricity,
            props.extent,
            props.solidity,
        ],
        dtype=np.float32,
    )


def extract_features(
    img_bgr: np.ndarray,
    binary_mask: np.ndarray | None = None,
    feature_groups: FeatureGroups | str = "all",
    segmented: bool = False,
) -> np.ndarray:
    """
    Extract and concatenate feature vector.

    With segmentation + 'all': 220 dim (192+9+4+10+5)
    Without segmentation + color+texture: 215 dim
    """
    groups = feature_groups.lower() if isinstance(feature_groups, str) else feature_groups
    parts: list[np.ndarray] = []

    if groups in ("color", "all"):
        parts.append(extract_color_histogram(img_bgr, segmented))
        parts.append(extract_color_moments(img_bgr, segmented))

    if groups in ("texture", "all"):
        parts.append(extract_glcm_features(img_bgr, segmented))
        parts.append(extract_lbp_features(img_bgr, segmented))

    if groups == "shape":
        if not segmented or binary_mask is None:
            raise ValueError("Shape features require segmentation and binary_mask.")
        shape = extract_shape_features(binary_mask)
        if shape is None:
            shape = np.zeros(5, dtype=np.float32)
        parts.append(shape)
    elif groups == "all" and segmented and binary_mask is not None:
        shape = extract_shape_features(binary_mask)
        if shape is None:
            shape = np.zeros(5, dtype=np.float32)
        parts.append(shape)

    if not parts:
        raise ValueError(f"Unknown feature_groups: {feature_groups}")

    return np.concatenate(parts)


def expected_feature_dim(feature_groups: FeatureGroups | str, segmented: bool) -> int:
    """Return expected feature vector dimension."""
    groups = feature_groups.lower()
    dim = 0
    if groups in ("color", "all"):
        dim += 192 + 9
    if groups in ("texture", "all"):
        dim += 4 + 10
    if groups == "shape" or (groups == "all" and segmented):
        dim += 5
    return dim


def get_feature_group_names(feature_groups: FeatureGroups | str, segmented: bool) -> list[str]:
    """Names for feature importance mapping."""
    groups = feature_groups.lower()
    names: list[str] = []
    if groups in ("color", "all"):
        names.extend([f"hist_{i}" for i in range(192)])
        names.extend(["mean_h", "std_h", "skew_h", "mean_s", "std_s", "skew_s", "mean_v", "std_v", "skew_v"])
    if groups in ("texture", "all"):
        names.extend(["glcm_contrast", "glcm_energy", "glcm_homogeneity", "glcm_correlation"])
        names.extend([f"lbp_{i}" for i in range(10)])
    if groups == "shape" or (groups == "all" and segmented):
        names.extend(["area", "perimeter", "eccentricity", "extent", "solidity"])
    return names
