"""Smoke tests for the full pipeline: process_image, batch_extract_features,
and run_classical_scenario on synthetic sample data.

These tests catch the class of bugs that reach Kaggle runtime:
- predictions_s5 IndexError (test_v misused as iloc index)
- restoration toggle not propagating through the pipeline
- cache validation not working across different splits
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.pipeline import process_image
from src.preprocessing import preprocess_from_array

# ---------------------------------------------------------------------------
# process_image - restoration propagation
# ---------------------------------------------------------------------------

def _solid_image(color: tuple[int, int, int] = (120, 80, 60)) -> np.ndarray:
    img = np.full((60, 60, 3), color, dtype=np.uint8)
    return img


def test_process_image_restoration_ssr_differs_from_none():
    """restoration='ssr' and 'none' must produce different pixel values."""
    img = _solid_image()
    out_ssr = process_image(img_bgr=img, restoration="ssr", enhancement="none")
    out_raw = process_image(img_bgr=img, restoration="none", enhancement="none")
    assert out_ssr["img"] is not None
    assert out_raw["img"] is not None
    assert not np.array_equal(out_ssr["img"], out_raw["img"]), (
        "SSR and raw outputs must differ - restoration flag had no effect"
    )


def test_process_image_restoration_none_matches_preprocess_raw():
    """restoration='none' in process_image must match preprocess_from_array(apply_restoration=False)."""
    img = _solid_image()
    out = process_image(img_bgr=img, restoration="none", enhancement="none")
    expected = preprocess_from_array(img, apply_restoration=False)
    np.testing.assert_array_equal(out["img"], expected)


def test_process_image_segment_flag():
    """do_segment=True should set mask; do_segment=False should leave mask as None."""
    img = _solid_image()
    seg = process_image(img_bgr=img, restoration="ssr", do_segment=True)
    noseg = process_image(img_bgr=img, restoration="ssr", do_segment=False)
    assert seg["mask"] is not None
    assert noseg["mask"] is None


# ---------------------------------------------------------------------------
# run_classical_scenario - end-to-end smoke on synthetic data
# (catches the predictions_s5 IndexError class of bug)
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_dataset(tmp_path: Path):
    """Create 12 real (tiny) images in a temp dir and return train/val/test DataFrames."""
    rng = np.random.default_rng(0)
    records = []
    for commodity in ("Apple", "Banana"):
        for label in ("fresh", "rotten"):
            for idx in range(3):
                folder = tmp_path / f"{commodity}__{label.capitalize()}"
                folder.mkdir(parents=True, exist_ok=True)
                fp = folder / f"{idx:03d}.jpg"
                img = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
                cv2.imwrite(str(fp), img)
                records.append({"filepath": str(fp), "label": label, "commodity": commodity})

    df = pd.DataFrame(records)
    df["strat_key"] = df["commodity"] + "_" + df["label"]
    # 8 train / 2 val / 2 test (manually split for determinism)
    train = df.iloc[:8].reset_index(drop=True)
    val   = df.iloc[8:10].reset_index(drop=True)
    test  = df.iloc[10:].reset_index(drop=True)
    return train, val, test, tmp_path


def test_run_classical_scenario_s1_produces_csv_and_no_indexerror(tiny_dataset, tmp_path):
    """S1 (raw baseline, no seg) must complete without IndexError and write metrics CSV."""
    from src.experiments import run_classical_scenario

    train, val, test, _ = tiny_dataset
    metrics_dir = tmp_path / "metrics"
    figures_dir = tmp_path / "figures"
    models_dir  = tmp_path / "models"
    cache_dir   = tmp_path / "cache"
    for d in (metrics_dir, figures_dir, models_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    result = run_classical_scenario(
        scenario_id=1,
        train_df=train, val_df=val, test_df=test,
        metrics_dir=metrics_dir,
        figures_dir=figures_dir,
        models_dir=models_dir,
        cache_dir=cache_dir,
    )

    assert "test_metrics" in result
    assert "f1_weighted" in result["test_metrics"]
    csv = metrics_dir / "scenario_01.csv"
    assert csv.exists(), "scenario_01.csv should be written by run_classical_scenario"
    df_csv = pd.read_csv(csv)
    assert "restoration" in df_csv.columns
    assert df_csv.loc[0, "restoration"] == "none"


def test_run_classical_scenario_s5_predictions_csv_aligned(tiny_dataset, tmp_path):
    """S5 must write predictions_s5.csv without IndexError and with correct column count."""
    from src.experiments import run_classical_scenario

    train, val, test, _ = tiny_dataset
    metrics_dir = tmp_path / "metrics"
    figures_dir = tmp_path / "figures"
    models_dir  = tmp_path / "models"
    cache_dir   = tmp_path / "cache"
    for d in (metrics_dir, figures_dir, models_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Write best_enhancement.txt so S5's "best" resolves cleanly
    (metrics_dir / "best_enhancement.txt").write_text("none", encoding="utf-8")

    result = run_classical_scenario(
        scenario_id=5,
        train_df=train, val_df=val, test_df=test,
        metrics_dir=metrics_dir,
        figures_dir=figures_dir,
        models_dir=models_dir,
        cache_dir=cache_dir,
    )

    pred_csv = metrics_dir / "predictions_s5.csv"
    assert pred_csv.exists(), "predictions_s5.csv must be written for S5"
    pred_df = pd.read_csv(pred_csv)
    # y_pred must align with rows in the CSV (regression test for old .iloc[DataFrame] bug)
    assert "pred" in pred_df.columns
    assert len(pred_df) == len(result["y_pred"]), (
        "predictions_s5.csv row count must match y_pred length - "
        "mismatch indicates the old test_df.iloc[test_v] bug is back"
    )
