"""Image processing pipeline compositor for notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from src.enhancement import apply_enhancement
from src.features import extract_features
from src.preprocessing import preprocess_from_array, preprocess_pipeline, to_uint8
from src.segmentation import segment_fruit


def process_image(
    path: str | Path | None = None,
    img_bgr: np.ndarray | None = None,
    restoration: str = "ssr",
    enhancement: str = "none",
    do_segment: bool = False,
    feature_groups: str = "all",
    extract_feat: bool = False,
) -> dict[str, Any]:
    """
    Full pipeline: preprocess -> (restore) -> enhance -> (segment) -> (features).

    Stages:
        - restoration: "ssr" applies Single-Scale Retinex illumination
          correction; "none" gives a true raw baseline (resize only).
        - enhancement: "none"/"clahe"/"histeq"/"gamma" on top of restoration.

    Provide either path or img_bgr.

    Returns dict with keys:
        img, mask, object_ratio, used_fallback, features (if extract_feat)
    """
    apply_restoration = str(restoration).lower() == "ssr"
    if path is not None:
        img = preprocess_pipeline(path, apply_restoration=apply_restoration)
        if img is None:
            return {"img": None, "mask": None, "object_ratio": 0.0, "used_fallback": True, "features": None}
    elif img_bgr is not None:
        img = preprocess_from_array(img_bgr, apply_restoration=apply_restoration)
    else:
        raise ValueError("Provide either path or img_bgr.")

    img = apply_enhancement(img, enhancement)

    mask = None
    object_ratio = 1.0
    used_fallback = False

    if do_segment:
        img, mask, object_ratio, used_fallback = segment_fruit(img)

    result: dict[str, Any] = {
        "img": img,
        "mask": mask,
        "object_ratio": object_ratio,
        "used_fallback": used_fallback,
        "features": None,
    }

    if extract_feat:
        result["features"] = extract_features(
            img, mask if do_segment else None, feature_groups, segmented=do_segment
        )

    return result


def image_to_cnn_input(img_bgr: np.ndarray) -> np.ndarray:
    """Convert uint8 BGR to float32 [-1,1] NHWC batch of 1 for MobileNetV2."""
    rgb = cv2.cvtColor(to_uint8(img_bgr), cv2.COLOR_BGR2RGB)
    normalized = (rgb.astype(np.float32) / 127.5) - 1.0
    return np.expand_dims(normalized, axis=0)


def log_segmentation_failure(
    filepath: str,
    commodity: str,
    label: str,
    object_ratio: float,
    log_path: Path,
) -> None:
    """Append one row to segmentation_failures.csv."""
    import pandas as pd

    row = pd.DataFrame(
        [{"filepath": filepath, "commodity": commodity, "label": label, "object_ratio": object_ratio}]
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        row.to_csv(log_path, mode="a", header=False, index=False)
    else:
        row.to_csv(log_path, index=False)


def batch_extract_features(
    filepaths: list[str],
    enhancement: str = "none",
    do_segment: bool = False,
    feature_groups: str = "all",
    show_progress: bool = True,
    metadata: "pd.DataFrame | None" = None,
    failure_log_path: Path | None = None,
    restoration: str = "ssr",
) -> tuple[np.ndarray, list[int]]:
    """
    Extract features for a list of image paths.
    Returns (X, valid_indices) - rows aligned with valid_indices into filepaths.
    """
    from tqdm import tqdm

    vectors: list[np.ndarray] = []
    valid_idx: list[int] = []
    iterator = tqdm(filepaths, desc="Extracting features") if show_progress else filepaths

    for i, fp in enumerate(iterator):
        out = process_image(
            path=fp,
            restoration=restoration,
            enhancement=enhancement,
            do_segment=do_segment,
            feature_groups=feature_groups,
            extract_feat=True,
        )
        if out["features"] is not None:
            vectors.append(out["features"])
            valid_idx.append(i)
            if out["used_fallback"] and failure_log_path is not None and metadata is not None:
                row = metadata.iloc[i]
                log_segmentation_failure(
                    filepath=fp,
                    commodity=str(row.get("commodity", "")),
                    label=str(row.get("label", "")),
                    object_ratio=float(out["object_ratio"]),
                    log_path=failure_log_path,
                )

    if not vectors:
        return np.empty((0, 0)), valid_idx
    return np.vstack(vectors), valid_idx
