"""Quick smoke test for src modules (no dataset required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.enhancement import apply_enhancement
from src.features import expected_feature_dim, extract_features
from src.preprocessing import apply_ssr, normalize_pixels, resize_image
from src.segmentation import segment_fruit


def main() -> None:
    rng = np.random.default_rng(42)
    img = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)

    resized = resize_image(img)
    assert resized.shape == (224, 224, 3)

    normed = normalize_pixels(resized)
    assert normed.max() <= 1.0

    ssr = apply_ssr(resized)
    assert ssr.dtype == np.uint8

    for method in ("none", "clahe", "histeq", "gamma"):
        out = apply_enhancement(ssr, method)
        assert out.shape == (224, 224, 3)

    masked, mask, ratio, fallback = segment_fruit(ssr)
    assert masked.shape == (224, 224, 3)

    feat_no_seg = extract_features(ssr, None, "all", segmented=False)
    assert len(feat_no_seg) == expected_feature_dim("all", segmented=False)

    feat_seg = extract_features(masked, mask, "all", segmented=True)
    assert len(feat_seg) == expected_feature_dim("all", segmented=True)

    print("Smoke test OK.")
    print(f"  Features without seg: {len(feat_no_seg)} dim")
    print(f"  Features with seg:    {len(feat_seg)} dim")
    print(f"  Object ratio: {ratio:.3f}, fallback: {fallback}")


if __name__ == "__main__":
    main()
