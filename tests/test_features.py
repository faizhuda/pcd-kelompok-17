"""Unit tests for src/features.py."""

import numpy as np
import pytest

from src.features import (
    expected_feature_dim,
    extract_color_histogram,
    extract_color_moments,
    extract_features,
    extract_glcm_features,
    extract_lbp_features,
    extract_shape_features,
    get_object_pixels,
)


def _make_img(h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _make_mask(h: int = 64, w: int = 64, fill: bool = True) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if fill:
        mask[16:48, 16:48] = 255
    return mask


def test_color_histogram_dim():
    img = _make_img()
    out = extract_color_histogram(img, segmented=False)
    assert out.shape == (192,)
    assert out.dtype == np.float32


def test_color_moments_dim():
    img = _make_img()
    out = extract_color_moments(img, segmented=False)
    assert out.shape == (9,)
    assert out.dtype == np.float32


def test_glcm_features_dim():
    img = _make_img()
    out = extract_glcm_features(img, segmented=False)
    assert out.shape == (4,)
    assert out.dtype == np.float32


def test_lbp_features_dim():
    img = _make_img()
    out = extract_lbp_features(img, segmented=False)
    assert out.shape == (10,)
    assert out.dtype == np.float32


def test_shape_features_dim():
    mask = _make_mask()
    out = extract_shape_features(mask)
    assert out is not None
    assert out.shape == (5,)


def test_shape_features_empty_mask_returns_none():
    empty_mask = np.zeros((64, 64), dtype=np.uint8)
    out = extract_shape_features(empty_mask)
    assert out is None


def test_feature_dim_without_segmentation():
    """215 dims: color(192+9) + texture(4+10), no shape."""
    img = _make_img()
    out = extract_features(img, binary_mask=None, feature_groups="all", segmented=False)
    assert out.shape == (215,)
    assert expected_feature_dim("all", segmented=False) == 215


def test_feature_dim_with_segmentation():
    """220 dims: color(192+9) + texture(4+10) + shape(5)."""
    img = _make_img()
    mask = _make_mask()
    out = extract_features(img, binary_mask=mask, feature_groups="all", segmented=True)
    assert out.shape == (220,)
    assert expected_feature_dim("all", segmented=True) == 220


def test_feature_dim_color_only():
    img = _make_img()
    out = extract_features(img, feature_groups="color", segmented=False)
    assert out.shape == (201,)  # 192 + 9
    assert expected_feature_dim("color", segmented=False) == 201


def test_feature_dim_texture_only():
    img = _make_img()
    out = extract_features(img, feature_groups="texture", segmented=False)
    assert out.shape == (14,)  # 4 + 10
    assert expected_feature_dim("texture", segmented=False) == 14


def test_feature_dim_shape_requires_mask():
    img = _make_img()
    with pytest.raises(ValueError, match="Shape features require segmentation"):
        extract_features(img, binary_mask=None, feature_groups="shape", segmented=False)


def test_get_object_pixels_no_segmentation():
    img = _make_img()
    pixels = get_object_pixels(img, segmented=False)
    assert pixels.shape == (64 * 64, 3)


def test_get_object_pixels_with_mask():
    img = _make_img()
    # Make part of image black (background)
    masked = img.copy()
    masked[:32, :] = 0
    pixels = get_object_pixels(masked, segmented=True)
    # Should return only non-black pixels
    assert pixels.shape[0] < 64 * 64


def test_get_object_pixels_fallback_when_mostly_black():
    """If < 5% pixels are non-black, fall back to full image."""
    img = _make_img()
    mostly_black = np.zeros_like(img)
    mostly_black[0, 0] = [10, 20, 30]  # tiny non-black area
    pixels = get_object_pixels(mostly_black, segmented=True)
    assert pixels.shape[0] == 64 * 64
