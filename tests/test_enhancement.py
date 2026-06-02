"""Unit tests for src/enhancement.py."""

import numpy as np
import pytest

from src.enhancement import apply_enhancement, clahe_enhance, gamma_adaptive, histeq_global


def _make_img(h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _make_dark_img() -> np.ndarray:
    """Very dark image — should trigger gamma > 1."""
    return np.full((64, 64, 3), 10, dtype=np.uint8)


def test_clahe_output_shape():
    img = _make_img()
    out = clahe_enhance(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_histeq_output_shape():
    img = _make_img()
    out = histeq_global(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_gamma_output_shape():
    img = _make_img()
    out = gamma_adaptive(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_gamma_brightens_dark_image():
    dark = _make_dark_img()
    out = gamma_adaptive(dark)
    # dark image should get brighter on average
    assert float(out.mean()) > float(dark.mean())


def test_apply_none_returns_uint8():
    img = _make_img()
    out = apply_enhancement(img, "none")
    assert out.dtype == np.uint8
    assert out.shape == img.shape


@pytest.mark.parametrize("method", ["clahe", "histeq", "gamma"])
def test_apply_enhancement_all_methods(method):
    img = _make_img()
    out = apply_enhancement(img, method)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_apply_enhancement_unknown_raises():
    img = _make_img()
    with pytest.raises(ValueError, match="Unknown enhancement method"):
        apply_enhancement(img, "unknown_method")


def test_apply_enhancement_accepts_float():
    img = _make_img().astype(np.float32) / 255.0
    out = apply_enhancement(img, "none")
    assert out.dtype == np.uint8
