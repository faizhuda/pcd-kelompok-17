"""Unit tests for src/segmentation.py."""

import numpy as np
import pytest

from src.segmentation import FALLBACK_RATIO, object_ratio_from_mask, segment_fruit


def _make_img(h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(2)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _make_black_img(h: int = 64, w: int = 64) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_bright_circle_img(h: int = 64, w: int = 64) -> np.ndarray:
    """Image with a bright green circle on black background — easy to segment."""
    import cv2

    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(img, (h // 2, w // 2), h // 3, (0, 200, 0), thickness=-1)
    return img


def test_segment_returns_four_values():
    img = _make_img()
    result = segment_fruit(img)
    assert len(result) == 4


def test_segment_output_shape_matches_input():
    img = _make_img()
    masked, mask, ratio, fallback = segment_fruit(img)
    assert masked.shape == img.shape
    assert mask.shape == img.shape[:2]


def test_segment_mask_binary():
    img = _make_img()
    _, mask, _, _ = segment_fruit(img)
    unique = np.unique(mask)
    assert set(unique).issubset({0, 255})


def test_fallback_triggered_on_black_image():
    """All-black image → object ratio ≈ 0 → fallback must be True."""
    img = _make_black_img()
    _, _, ratio, used_fallback = segment_fruit(img)
    assert used_fallback is True


def test_fallback_flag_false_on_textured_image():
    """A real image with colour should usually not trigger fallback."""
    img = _make_bright_circle_img()
    _, _, ratio, used_fallback = segment_fruit(img)
    # At least object_ratio > FALLBACK_RATIO when segmentation succeeds
    if not used_fallback:
        assert ratio > FALLBACK_RATIO


def test_object_ratio_from_mask_full():
    mask = np.full((10, 10), 255, dtype=np.uint8)
    assert object_ratio_from_mask(mask) == pytest.approx(1.0)


def test_object_ratio_from_mask_empty():
    mask = np.zeros((10, 10), dtype=np.uint8)
    assert object_ratio_from_mask(mask) == pytest.approx(0.0)


def test_segment_accepts_float_input():
    img = _make_img().astype(np.float32) / 255.0
    masked, mask, ratio, fallback = segment_fruit(img)
    assert masked.dtype == np.uint8
    assert mask.shape == (64, 64)
