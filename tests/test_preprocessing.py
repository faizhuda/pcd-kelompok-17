"""Unit tests for src/preprocessing.py."""

import numpy as np

from src.preprocessing import (
    apply_ssr,
    normalize_pixels,
    preprocess_from_array,
    resize_image,
)


def _make_img(h: int = 100, w: int = 100) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_resize_output_shape():
    img = _make_img(50, 80)
    out = resize_image(img, (224, 224))
    assert out.shape == (224, 224, 3)


def test_resize_default_shape():
    img = _make_img(300, 400)
    out = resize_image(img)
    assert out.shape == (224, 224, 3)


def test_normalize_range():
    img = _make_img()
    out = normalize_pixels(img)
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_normalize_already_float_clips():
    img = np.full((10, 10, 3), 300, dtype=np.float32)
    out = normalize_pixels(img)
    assert out.max() <= 1.0


def test_ssr_preserves_shape():
    img = _make_img()
    out = apply_ssr(img)
    assert out.shape == img.shape


def test_ssr_output_dtype():
    img = _make_img()
    out = apply_ssr(img)
    assert out.dtype == np.uint8


def test_ssr_accepts_float_input():
    img = _make_img().astype(np.float32) / 255.0
    out = apply_ssr(img)
    assert out.dtype == np.uint8
    assert out.shape == (100, 100, 3)


def test_preprocess_from_array_shape():
    img = _make_img(300, 200)
    out = preprocess_from_array(img)
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8
